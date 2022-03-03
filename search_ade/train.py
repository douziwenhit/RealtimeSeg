import os
import sys
import time
import glob
import logging
from tqdm import tqdm
from random import shuffle

import torch
import torch.nn as nn
import torch.utils
from tensorboardX import SummaryWriter

import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from thop import profile

from config_search import config
from dataloader import get_train_loader
from tools.datasets import Cityscapes

from tools.utils.init_func import init_weight
from tools.seg_opr.loss_opr import ProbOhemCrossEntropy2d

from eval import SegEvaluator

from supernet import Architect
from tools.utils.darts_utils import create_exp_dir, save, plot_op, plot_path_width, objective_acc_lat
from model_search import Network_Multi_Path as Network
from model_seg import Network_Multi_Path_Infer
import torch.utils.data.distributed

import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed

from tensorboardX import SummaryWriter


logger = SummaryWriter('log')


config.gpu_devices = 0,1,2,3
gpu_devices = ','.join([str(id) for id in config.gpu_devices])
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices

def main():

    ngpus_per_node = torch.cuda.device_count()
    config.world_size = ngpus_per_node * config.world_size
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config))
        
        
def main_worker(gpu, ngpus_per_node, config):
    config.gpu = gpu
    ngpus_per_node = torch.cuda.device_count()
    print("Use GPU: {} for training".format(config.gpu))

    config.rank = config.rank * ngpus_per_node + gpu
    dist.init_process_group(backend=config.dist_backend, init_method=config.dist_url,
                            world_size=config.world_size, rank=config.rank)

    assert type(config.pretrain) == bool or type(config.pretrain) == str
    update_arch = True
    if config.pretrain == True:
        update_arch = False
    logging.info("args = %s", str(config))

    # preparation ################
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    seed = config.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # config network and criterion ################
    min_kept = int(config.batch_size * config.image_height * config.image_width // (16 * config.gt_down_sampling ** 2))
    ohem_criterion = ProbOhemCrossEntropy2d(ignore_label=255, thresh=0.7, min_kept=min_kept, use_weight=False).cuda()

    # Model #######################################
    print('==> Making model..')
    model = Network(config.num_classes, config.layers, ohem_criterion, Fch=config.Fch,
                    width_mult_list=config.width_mult_list, prun_modes=config.prun_modes,
                    stem_head_width=config.stem_head_width)
    torch.cuda.set_device(config.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.cuda(config.gpu)
    config.num_workers = int(config.num_workers / ngpus_per_node)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.gpu])
    flops, params = profile(model, inputs=(torch.randn(1, 3, 1024, 2048),), verbose=False)
    logging.info("params = %fMB, FLOPs = %fGB", params / 1e6, flops / 1e9)
    architect = Architect(model, config)

    if type(config.pretrain) == str:
        partial = torch.load(config.pretrain + "/weights.pt", map_location='cuda')
        state = model.state_dict()
        pretrained_dict = {k: v for k, v in partial.items() if k in state and state[k].size() == partial[k].size()}
        state.update(pretrained_dict)
        model.load_state_dict(state)
    else:
        init_weight(model, nn.init.kaiming_normal_, nn.BatchNorm2d, config.bn_eps, config.bn_momentum, mode='fan_in',
                    nonlinearity='relu')
        # Optimizer ###################################
    base_lr = config.lr
    parameters = []
    parameters += list(model.stem.parameters())
    parameters += list(model.cells.parameters())
    parameters += list(model.refine32.parameters())
    parameters += list(model.refine16.parameters())
    parameters += list(model.head0.parameters())
    parameters += list(model.head1.parameters())
    parameters += list(model.head2.parameters())
    parameters += list(model.head02.parameters())
    parameters += list(model.head12.parameters())
    optimizer = torch.optim.SGD(
        parameters,
        lr=base_lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay)

    # lr policy ##############################
    lr_policy = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.978)

    # data loader ###########################
    data_setting = {'img_root': config.img_root_folder,
                    'gt_root': config.gt_root_folder,
                    'train_source': config.train_source,
                    'eval_source': config.eval_source,
                    'down_sampling': config.down_sampling}
    index_select = list(range(config.num_train_imgs))
    shuffle(index_select)  # shuffle to make sure balanced dataset split
    train_loader_model = get_train_loader(config, Cityscapes, portion=config.train_portion, index_select=index_select)
    train_loader_arch = get_train_loader(config, Cityscapes, portion=config.train_portion - 1,
                                         index_select=index_select)

    tbar = tqdm(range(config.nepochs), ncols=80)
    valid_mIoU_history = [];
    FPSs_history = [];
    latency_supernet_history = [];
    latency_weight_history = [];
    valid_names = ["8s", "16s", "32s", "8s_32s", "16s_32s"]
    arch_names = {0: "teacher", 1: "student"}

    for epoch in tbar:
        logging.info(config.pretrain)
        logging.info(config.save)
        logging.info("lr: " + str(optimizer.param_groups[0]['lr']))

        # training
        tbar.set_description("[Epoch %d/%d][train...]" % (epoch + 1, config.nepochs))
        train(config.pretrain, train_loader_model, train_loader_arch, model, architect, ohem_criterion, optimizer, lr_policy,
              logger, epoch, update_arch=update_arch)
        torch.cuda.empty_cache()
        lr_policy.step()

    save(model, os.path.join(config.save, 'weights.pt'))
    if type(config.pretrain) == str:
        # contains arch_param names: {"alphas": alphas, "betas": betas, "gammas": gammas, "ratios": ratios}
        for idx, arch_name in enumerate(model._arch_names):
            state = {}
            for name in arch_name['alphas']:
                state[name] = getattr(model, name)
            for name in arch_name['betas']:
                state[name] = getattr(model, name)
            for name in arch_name['ratios']:
                state[name] = getattr(model, name)
            torch.save(state, os.path.join(config.save, "arch_%d_%d.pt" % (idx, epoch)))
            torch.save(state, os.path.join(config.save, "arch_%d.pt" % (idx)))


            
def train(pretrain, train_loader_model, train_loader_arch, model, architect, criterion, optimizer, lr_policy, logger, epoch, update_arch=True):
    model.train()

    bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
    pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout, bar_format=bar_format, ncols=80)
    dataloader_model = iter(train_loader_model)
    dataloader_arch = iter(train_loader_arch)

    for step in pbar:
        optimizer.zero_grad()

        minibatch = dataloader_model.next()
        imgs = minibatch['data']
        target = minibatch['label']
        imgs = imgs.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        if update_arch:
            # get a random minibatch from the search queue with replacement
            pbar.set_description("[Arch Step %d/%d]" % (step + 1, len(train_loader_model)))
            minibatch = dataloader_arch.next()
            imgs_search = minibatch['data']
            target_search = minibatch['label']
            imgs_search = imgs_search.cuda(non_blocking=True)
            target_search = target_search.cuda(non_blocking=True)
            loss_arch = architect.step(imgs, target, imgs_search, target_search)
            if (step+1) % 10 == 0:
                logger.add_scalar('loss_arch/train', loss_arch, epoch*len(pbar)+step)
                logger.add_scalar('arch/latency_supernet', architect.latency_supernet, epoch*len(pbar)+step)

        optimizer.zero_grad()
        loss = model._loss(imgs, target, pretrain)
        logger.add_scalar('loss/train', loss, epoch*len(pbar)+step)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        optimizer.zero_grad()

        pbar.set_description("[Step %d/%d]" % (step + 1, len(train_loader_model)))
    torch.cuda.empty_cache()
    # del loss
    # if update_arch: del loss_arch


    

if __name__=='__main__':
    main()
