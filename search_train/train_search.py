from __future__ import division

import os
import sys
import time
import glob
import logging
from tqdm import tqdm
from random import shuffle
search = True
train = False
import torch
import torch.nn as nn
import torch.utils
from tensorboardX import SummaryWriter
import seg_metrics

import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from thop import profile
if search == True:
    from config import config
    from dataloader import get_train_loader
    from datasets import Cityscapes

    from utils.init_func import init_weight
    from seg_opr.loss_opr import ProbOhemCrossEntropy2d
    from eval import SegEvaluator

    from supernet import Architect
    from utils.darts_utils import create_exp_dir, save, plot_op, plot_path_width, objective_acc_lat
    from model_search import Network_Multi_Path as Network
    from model_seg import Network_Multi_Path_Infer

if train == True:
    import os
    import sys
    import time
    import glob
    import logging
    from tqdm import tqdm

    import torch
    import torch.nn as nn
    import torch.utils
    import torch.nn.functional as F
    from tensorboardX import SummaryWriter

    import numpy as np
    from thop import profile

    from config_train import config

    if config.is_eval:
        config.save = 'eval-{}-{}'.format(config.save, time.strftime("%Y%m%d-%H%M%S"))
    else:
        config.save = 'train-{}-{}'.format(config.save, time.strftime("%Y%m%d-%H%M%S"))
    from dataloader import get_train_loader
    from datasets import Cityscapes

    import argparse
    from tensorboardX import SummaryWriter

    from utils.init_func import init_weight
    from seg_opr.loss_opr import ProbOhemCrossEntropy2d
    from eval import SegEvaluator
    from test import SegTester
    import segmentation_models_pytorch as smp
    from utils.darts_utils import create_exp_dir, save, plot_op, plot_path_width, objective_acc_lat
    from model_seg import Network_Multi_Path_Infer as Network
    import seg_metrics

    import torch.distributed as dist
    import torch.multiprocessing as mp
    import torch.utils.data.distributed
#######muilt_gpu train################################################
    parser = argparse.ArgumentParser(description='cifar10 classification models')
    parser.add_argument('--lr', default=0.1, help='')
    parser.add_argument('--resume', default=None, help='')
    parser.add_argument('--batch_size', type=int, default=768, help='')
    parser.add_argument('--num_workers', type=int, default=4, help='')
    parser.add_argument("--gpu_devices", type=int, nargs='+', default=None, help="")

    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:4596', type=str, help='')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='')
    parser.add_argument('--rank', default=0, type=int, help='')
    parser.add_argument('--world_size', default=1, type=int, help='')
    parser.add_argument('--distributed', action='store_true', help='')
    args = parser.parse_args()


    args.gpu_devices =0,
    gpu_devices = ','.join([str(id) for id in args.gpu_devices])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
    config.gpu_devices = gpu_devices


def adjust_learning_rate(base_lr, power, optimizer, epoch, total_epoch):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * power

def main(search=True,train=False):
    if search == True:
        search_train(pretrain=config.pretrain)
    if train == True:
        config.save = 'search-{}-{}'.format(config.save, time.strftime("%Y%m%d-%H%M%S"))
        create_exp_dir(config.save, scripts_to_save=glob.glob('*.py') + glob.glob('*.sh'))
        args = parser.parse_args()
        ngpus_per_node = torch.cuda.device_count()
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))

def search_train(pretrain=True):
    config.save = 'search-{}-{}'.format(config.save, time.strftime("%Y%m%d-%H%M%S"))
    create_exp_dir(config.save, scripts_to_save=glob.glob('*.py') + glob.glob('*.sh'))
    logger = SummaryWriter(config.save)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(config.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    assert type(pretrain) == bool or type(pretrain) == str
    update_arch = True
    if pretrain == True:
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
    ohem_criterion = ProbOhemCrossEntropy2d(ignore_label=255, thresh=0.7, min_kept=min_kept, use_weight=False)

    # Model #######################################
    model = Network(config.num_classes, config.layers, ohem_criterion, Fch=config.Fch,
                    width_mult_list=config.width_mult_list, prun_modes=config.prun_modes,
                    stem_head_width=config.stem_head_width)

    model = model.cuda()
    if type(pretrain) == str:
        partial = torch.load("/home/wangshuo/douzi/FasterSeg/dist_latency/pretrain-256x512_F12.L16_batch1/weights.pt",
                             map_location='cuda:0')
        state = model.state_dict()
        pretrained_dict = {k: v for k, v in partial.items() if k in state and state[k].size() == partial[k].size()}
        state.update(pretrained_dict)
        model.load_state_dict(state)
    else:
        init_weight(model, nn.init.kaiming_normal_, nn.BatchNorm2d, config.bn_eps, config.bn_momentum, mode='fan_in',
                    nonlinearity='relu')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    architect = Architect(model, config)
    # flops, params = profile(model, inputs=(torch.randn(1, 3, 1024, 2048),), verbose=False)
    # logging.info("params = %fMB, FLOPs = %fGB", params / 1e6, flops / 1e9)
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

    evaluator = SegEvaluator(Cityscapes(data_setting, 'val', None), config.num_classes, config.image_mean,
                             config.image_std, model, config.eval_scale_array, config.eval_flip, 0, config=config,
                             verbose=False, save_path=None, show_image=False)

    if update_arch:
        logger.add_scalar("arch/latency_weight", config.latency_weight, 0)
        logging.info("arch_latency_weight = " + str(config.latency_weight))
        logger.add_scalar("arch/flops_weight", config.flops_weight, 0)
        logging.info("arch_flops_weight = " + str(config.flops_weight))

    tbar = tqdm(range(config.nepochs), ncols=80)
    valid_mIoU_history = [];
    FPSs_history = [];
    latency_supernet_history = [];
    latency_weight_history = [];
    valid_names = ["8s", "16s", "32s", "8s_32s", "16s_32s"]
    arch_names = {0: "teacher", 1: "student"}
    for epoch in tbar:
        logging.info(pretrain)
        logging.info(config.save)
        logging.info("lr: " + str(optimizer.param_groups[0]['lr']))

        logging.info("update arch: " + str(update_arch))

        # training
        tbar.set_description("[Epoch %d/%d][train...]" % (epoch + 1, config.nepochs))
        train(pretrain, train_loader_model, train_loader_arch, model, architect, ohem_criterion, optimizer, lr_policy,
              logger, epoch, update_arch=update_arch)
        torch.cuda.empty_cache()
        lr_policy.step()

        # validation
        tbar.set_description("[Epoch %d/%d][validation...]" % (epoch + 1, config.nepochs))
        with torch.no_grad():
            if pretrain == True:
                model.prun_mode = "min"
                valid_mIoUs = infer(epoch, model, evaluator, logger, FPS=False)
                for i in range(5):
                    logger.add_scalar('mIoU/val_min_%s' % valid_names[i], valid_mIoUs[i], epoch)
                    logging.info("Epoch %d: valid_mIoU_min_%s %.3f" % (epoch, valid_names[i], valid_mIoUs[i]))
                if len(model._width_mult_list) > 1:
                    model.prun_mode = "max"
                    valid_mIoUs = infer(epoch, model, evaluator, logger, FPS=False)
                    for i in range(5):
                        logger.add_scalar('mIoU/val_max_%s' % valid_names[i], valid_mIoUs[i], epoch)
                        logging.info("Epoch %d: valid_mIoU_max_%s %.3f" % (epoch, valid_names[i], valid_mIoUs[i]))
                    model.prun_mode = "random"
                    valid_mIoUs = infer(epoch, model, evaluator, logger, FPS=False)
                    for i in range(5):
                        logger.add_scalar('mIoU/val_random_%s' % valid_names[i], valid_mIoUs[i], epoch)
                        logging.info("Epoch %d: valid_mIoU_random_%s %.3f" % (epoch, valid_names[i], valid_mIoUs[i]))
            else:
                valid_mIoUss = [];
                FPSs = []
                model.prun_mode = None
                for idx in range(len(model._arch_names)):
                    valid_mIoUs, fps0, fps1 = infer(epoch, model, evaluator, logger)
                    valid_mIoUss.append(valid_mIoUs)
                    FPSs.append([fps0, fps1])
                    for i in range(5):
                        # preds
                        logger.add_scalar('mIoU/val_%s_%s' % (arch_names[idx], valid_names[i]), valid_mIoUs[i], epoch)
                        logging.info("Epoch %d: valid_mIoU_%s_%s %.3f" % (
                        epoch, arch_names[idx], valid_names[i], valid_mIoUs[i]))
                    if config.latency_weight > 0:
                        logger.add_scalar('Objective/val_%s_8s_32s' % arch_names[idx],
                                          objective_acc_lat(valid_mIoUs[3], 1000. / fps0), epoch)
                        logging.info("Epoch %d: Objective_%s_8s_32s %.3f" % (
                        epoch, arch_names[idx], objective_acc_lat(valid_mIoUs[3], 1000. / fps0)))
                        logger.add_scalar('Objective/val_%s_16s_32s' % arch_names[idx],
                                          objective_acc_lat(valid_mIoUs[4], 1000. / fps1), epoch)
                        logging.info("Epoch %d: Objective_%s_16s_32s %.3f" % (
                        epoch, arch_names[idx], objective_acc_lat(valid_mIoUs[4], 1000. / fps1)))
                valid_mIoU_history.append(valid_mIoUss)
                FPSs_history.append(FPSs)
                if update_arch:
                    latency_supernet_history.append(architect.latency_supernet)
                latency_weight_history.append(architect.latency_weight)
        save(model, os.path.join(config.save, 'weights.pt'))
        if type(pretrain) == str:
           
            for idx, arch_name in enumerate(model._arch_names):
                state = {}
                for name in arch_name['fais']:
                    state[name] = getattr(model, name)
                for name in arch_name['mjus']:
                    state[name] = getattr(model, name)
                for name in arch_name['thetas']:
                    state[name] = getattr(model, name)
                state["mIoU02"] = valid_mIoUs[3]
                state["mIoU12"] = valid_mIoUs[4]
                if pretrain is not True:
                    state["latency02"] = 1000. / fps0
                    state["latency12"] = 1000. / fps1
                torch.save(state, os.path.join(config.save, "arch_%d.pt" % epoch))
                torch.save(state, os.path.join(config.save, "arch.pt"))
        if update_arch:
            if config.latency_weight > 0:
                if (int(FPSs[0] >= config.FPS_max) + int(FPSs[1] >= config.FPS_max)) >= 1:
                    architect.latency_weight = architect.latency_weight / 2
                elif (int(FPSs[0] <= config.FPS_min) + int(FPSs[1] <= config.FPS_min)) > 0:
                    architect.latency_weight = architect.latency_weight * 2
            logger.add_scalar("arch/latency_weight", architect.latency_weight, epoch + 1)
            logging.info("arch_latency_weight = " + str(architect.latency_weight))
            if config.flops_weight > 0:
                if (int(FPSs[0] >= config.FPS_max) + int(FPSs[1] >= config.FPS_max)) >= 1:
                    architect.flops_weight /= 2
                elif (int(FPSs[0] <= config.FPS_min) + int(FPSs[1] <= config.FPS_min)) > 0:
                    architect.flops_weight *= 2
                logger.add_scalar("arch/latency_weight", architect.flops_weight, epoch + 1)
                logging.info("arch_latency_weight = " + str(architect.flops_weight))


def main_worker(gpu, ngpus_per_node, args):
    logger = SummaryWriter(config.save)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(config.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info("args = %s", str(config))
    # dist###########################
    args.gpu = gpu
    ngpus_per_node = torch.cuda.device_count()
    print("Use GPU: {} for training".format(args.gpu))

    args.rank = args.rank * ngpus_per_node + gpu
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)

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
    ohem_criterion = ProbOhemCrossEntropy2d(ignore_label=255, thresh=0.7, min_kept=min_kept, use_weight=False)
    distill_criterion = nn.KLDivLoss().cuda()

    # data loader ###########################
    if config.is_test:
        data_setting = {'img_root': config.img_root_folder,
                        'gt_root': config.gt_root_folder,
                        'train_source': config.train_eval_source,
                        'eval_source': config.eval_source,
                        'test_source': config.test_source,
                        'down_sampling': config.down_sampling}
    else:
        data_setting = {'img_root': config.img_root_folder,
                        'gt_root': config.gt_root_folder,
                        'train_source': config.train_source,
                        'eval_source': config.eval_source,
                        'test_source': config.test_source,
                        'down_sampling': config.down_sampling}

    train_loader = get_train_loader(config, Cityscapes)

    # Model #######################################
    models = []
    evaluators = []
    testers = []
    lasts = []
    start_epoch = -1

    for idx, arch_idx in enumerate(config.arch_idx):
        if config.load_epoch == "last":
            state = torch.load(os.path.join(config.load_path, "arch_%d.pt" % arch_idx), map_location='cuda:0')
        else:
            state = torch.load(os.path.join(config.load_path, "arch_%d_%d.pt" % (arch_idx, int(config.load_epoch))))

        model = Network(
            [state["alpha_%d_0" % arch_idx].detach(), state["alpha_%d_1" % arch_idx].detach(),
             state["alpha_%d_2" % arch_idx].detach()],
            [None, state["beta_%d_1" % arch_idx].detach(), state["beta_%d_2" % arch_idx].detach()],
            [state["ratio_%d_0" % arch_idx].detach(), state["ratio_%d_1" % arch_idx].detach(),
             state["ratio_%d_2" % arch_idx].detach()],
            num_classes=config.num_classes, layers=config.layers, Fch=config.Fch,
            width_mult_list=config.width_mult_list, stem_head_width=config.stem_head_width[idx])
        torch.cuda.set_device(args.rank)
        model.cuda(args.rank)
        config.num_workers = int(config.num_workers / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.rank], find_unused_parameters=True)

        mIoU02 = state["mIoU02"];
        latency02 = state["latency02"];
        obj02 = objective_acc_lat(mIoU02, latency02)
        mIoU12 = state["mIoU12"];
        latency12 = state["latency12"];
        obj12 = objective_acc_lat(mIoU12, latency12)
        if obj02 > obj12:
            last = [2, 0]
        else:
            last = [2, 1]
        lasts.append(last)
        model.module.build_structure(last)
        logging.info("net: " + str(model))
        for b in last:
            if len(config.width_mult_list) > 1:
                plot_op(getattr(model.module, "ops%d" % b), getattr(model.module, "path%d" % b),
                        width=getattr(model.module, "widths%d" % b), head_width=config.stem_head_width[idx][1],
                        F_base=config.Fch).savefig(os.path.join(config.save, "ops_%d_%d.png" % (arch_idx, b)),
                                                   bbox_inches="tight")
            else:
                plot_op(getattr(model.module, "ops%d" % b), getattr(model.module, "path%d" % b),
                        F_base=config.Fch).savefig(os.path.join(config.save, "ops_%d_%d.png" % (arch_idx, b)),
                                                   bbox_inches="tight")
        plot_path_width(model.module.lasts, model.module.paths, model.module.widths).savefig(
            os.path.join(config.save, "path_width%d.png" % arch_idx))
        plot_path_width([2, 1, 0], [model.module.path2, model.module.path1, model.module.path0],
                        [model.module.widths2, model.module.widths1, model.module.widths0]).savefig(
            os.path.join(config.save, "path_width_all%d.png" % arch_idx))
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('The number of parameters of model is', num_params)
        # logging.info("ops:" + str(model.module.mops))
        # logging.info("path:" + str(model.module.paths))
        # logging.info("last:" + str(model.module.lasts))
        model = model.cuda(args.rank)
        init_weight(model, nn.init.kaiming_normal_, torch.nn.BatchNorm2d, config.bn_eps, config.bn_momentum,
                    mode='fan_in', nonlinearity='relu')
        if arch_idx == 0 and len(config.arch_idx) > 1:
            model = smp.Unet(encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                             encoder_weights="imagenet",
                             # use `imagenet` pre-trained weights for encoder initialization
                             in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                             classes=19,  # model output channels (number of classes in your dataset)
                             ).cuda()

        elif config.is_eval:
            partial = torch.load(os.path.join(config.eval_path, "weights%d.pt" % arch_idx), map_location='cuda:0')
            state = model.module.state_dict()
            pretrained_dict = {k: v for k, v in partial.items() if k in state}
            state.update(pretrained_dict)
            model.module.load_state_dict(state)

        evaluator = SegEvaluator(Cityscapes(data_setting, 'val', None), config.num_classes, config.image_mean,
                                 config.image_std, model, config.eval_scale_array, config.eval_flip, 0, out_idx=0,
                                 config=config,
                                 verbose=False, save_path=None, show_image=False, show_prediction=False)
        evaluators.append(evaluator)
        tester = SegTester(Cityscapes(data_setting, 'test', None), config.num_classes, config.image_mean,
                           config.image_std, model, config.eval_scale_array, config.eval_flip, 0, out_idx=0,
                           config=config,
                           verbose=False, save_path=None, show_prediction=False)
        testers.append(tester)

        # Optimizer ###################################
        base_lr = config.lr
        if arch_idx == 1 or len(config.arch_idx) == 1:
            # optimize teacher solo OR student (w. distill from teacher)
            optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=config.momentum,
                                        weight_decay=config.weight_decay)
        models.append(model)

        if config.RESUME:
            path_checkpoint = "./checkpoint/ckpt_best_17.pth"  # 断点路径

            checkpoint = torch.load(path_checkpoint)  # 加载断点
            start_epoch = checkpoint['epoch']  # 设置开始的epoch
            if arch_idx == 1 or len(config.arch_idx) == 1:
                optimizer.load_state_dict(checkpoint['optimizer'])
                partial = torch.load(os.path.join(
                    '/home/wangshuo/douzi/pytorch-multigpu/train_dist/train-512x1024_student_batch20-20211128-202309',
                    "weights%d.pt" % arch_idx), map_location='cuda:0')
                state = model.module.state_dict()
                pretrained_dict = {k: v for k, v in partial.items() if k in state}
                state.update(pretrained_dict)
                model.module.load_state_dict(state)

    # Cityscapes ###########################################
    if config.is_eval:
        logging.info(config.load_path)
        logging.info(config.eval_path)
        logging.info(config.save)
        with torch.no_grad():
            if config.is_test:
                # test
                print("[test...]")
                with torch.no_grad():
                    test_student(0, models, testers, logger)
            else:
                # validation
                print("[validation...]")
                valid_mIoUs = infer_student(models, evaluators, logger)
                for idx, arch_idx in enumerate(config.arch_idx):
                    if arch_idx == 1:
                        logger.add_scalar("mIoU/val_student", valid_mIoUs[idx], 0)
                        logging.info("student's valid_mIoU %.3f" % (valid_mIoUs[idx]))
        exit(0)
    vaild_student_max = 0
    tbar = tqdm(range(config.nepochs), ncols=80)
    for epoch in range(start_epoch + 1, 300):
        logging.info(config.load_path)
        logging.info(config.save)
        logging.info("lr: " + str(optimizer.param_groups[0]['lr']))
        # training
        tbar.set_description("[Epoch %d/%d][train...]" % (epoch + 1, config.nepochs))
        train_mIoUs = train_student(train_loader, models, ohem_criterion, distill_criterion, optimizer, logger, epoch)
        torch.cuda.empty_cache()
        for idx, arch_idx in enumerate(config.arch_idx):
            if arch_idx == 1:
                logger.add_scalar("mIoU/train_student", train_mIoUs[idx], epoch)
                logging.info("student's train_mIoU %.3f" % (train_mIoUs[idx]))
        adjust_learning_rate(base_lr, 0.992, optimizer, epoch + 1, config.nepochs)

        # validation
        if not config.is_test and ((epoch + 1) % 1 == 0 or epoch == 0):
            tbar.set_description("[Epoch %d/%d][validation...]" % (epoch + 1, config.nepochs))
            with torch.no_grad():
                valid_mIoUs = infer_student(models, evaluators, logger)
                for idx, arch_idx in enumerate(config.arch_idx):
                    if arch_idx == 1:
                        logger.add_scalar("mIoU/val_student", valid_mIoUs[idx], epoch)
                        logging.info("student's valid_mIoU %.3f" % (valid_mIoUs[idx]))
                    save(models[1], os.path.join(config.save, "weights%d.pt" % 1))
                    # resume
                    checkpoint = {'optimizer': optimizer.state_dict(),
                                  'epoch': epoch}
                    torch.save(checkpoint, './checkpoint/ckpt_best_%d.pth' % epoch)
        # test
        if config.is_test and (epoch + 1) >= 250 and (epoch + 1) % 10 == 0:
            tbar.set_description("[Epoch %d/%d][test...]" % (epoch + 1, config.nepochs))
            with torch.no_grad():
                test_student(epoch, models, testers, logger)

        save(models[1], os.path.join(config.save, "weights%d.pt" % 1))


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
                logger.add_scalar('arch/flops_supernet', architect.flops_supernet, epoch * len(pbar) + step)
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


def infer(epoch, model, evaluator, logger, FPS=True):
    model.eval()
    mIoUs = []
    for idx in range(5):
        evaluator.out_idx = idx
        # _, mIoU = evaluator.run_online()
        _, mIoU = evaluator.run_online_multiprocess()
        mIoUs.append(mIoU)
    if FPS:
        fps0, fps1 = arch_logging(model, config, logger, epoch)
        return mIoUs, fps0, fps1
    else:
        return mIoUs
def train_student(train_loader, models, criterion, distill_criterion, optimizer, logger, epoch):
    if len(models) == 1:
        # train teacher solo
        models[0].train()
        models[0].cuda()
    else:
        # train student (w. distill from teacher)
        models[0].eval()
        models[1].train()

    bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
    pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout, bar_format=bar_format, ncols=80)
    dataloader = iter(train_loader)

    metrics = [ seg_metrics.Seg_Metrics(n_classes=config.num_classes) for _ in range(len(models)) ]
    lamb = 0.2
    for step in pbar:
        optimizer.zero_grad()

        minibatch = dataloader.next()
        imgs = minibatch['data']
        imgs = imgs.type(torch.FloatTensor)
        target = minibatch['label']
        #imgs = imgs.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        logits_list = []
        #hardTarget
        loss = 0
        #softTarget
        loss_kl = 0
        description = ""
        for idx, arch_idx in enumerate(config.arch_idx):
            model = models[idx]
            if arch_idx == 0 and len(models) > 1:
                with torch.no_grad():
                    imgs = imgs.cuda(non_blocking=True)
                    logits8 = model(imgs)
                    logits_list.append(logits8)
            else:
                imgs = imgs.cuda(non_blocking=True)
                logits8, logits16, logits32 = model(imgs)
                logits_list.append(logits8)
                logits8.cuda()
                logits16.cuda()
                logits32.cuda()
                loss = loss + criterion(logits8, target)
                loss = loss + lamb * criterion(logits16, target)
                loss = loss + lamb * criterion(logits32, target)
                loss.cuda()
                if len(logits_list) > 1:
                    distill_criterion = distill_criterion.cuda()
                    loss = loss + distill_criterion(F.softmax(logits_list[1], dim=1).log().cuda(), F.softmax(logits_list[0], dim=1).cuda())

            metrics[idx].update(logits8.data, target)
        description += "[mIoU%d: %.3f]"%(1, metrics[1].get_scores())

        pbar.set_description("[Step %d/%d]"%(step + 1, len(train_loader)) + description)
        logger.add_scalar('loss/train', loss+loss_kl, epoch*len(pbar)+step)

        loss.backward()
        optimizer.step()

    return [ metric.get_scores() for metric in metrics ]

def infer_student(models, evaluators, logger):
    mIoUs = []
    for model, evaluator in zip(models, evaluators):
        model.eval()
        _, mIoU = evaluator.run_online()
        #_, mIoU = evaluator.run_online_multiprocess()
        mIoUs.append(mIoU)
    return mIoUs

def test_student(epoch, models, testers, logger):
    for idx, arch_idx in enumerate(config.arch_idx):
        if arch_idx == 0: continue
        model = models[idx]
        tester = testers[idx]
        os.system("mkdir %s"%os.path.join(os.path.join(os.path.realpath('.'), config.save, "test")))
        model.eval()
        tester.run_online()
        os.system("mv %s %s"%(os.path.join(os.path.realpath('.'), config.save, "test"), os.path.join(os.path.realpath('.'), config.save, "test_%d_%d"%(arch_idx, epoch))))

def arch_logging(model, args, logger, epoch):
    input_size = (1, 3, 1024, 2048)
    net = Network_Multi_Path_Infer(
        [getattr(model, model._arch_names[model.arch_idx]["fais"][0]).clone().detach(), getattr(model, model._arch_names[model.arch_idx]["fais"][1]).clone().detach(), getattr(model, model._arch_names[model.arch_idx]["fais"][2]).clone().detach()],
        [None, getattr(model, model._arch_names[model.arch_idx]["mjus"][0]).clone().detach(), getattr(model, model._arch_names[model.arch_idx]["mjus"][1]).clone().detach()],
        [getattr(model, model._arch_names[model.arch_idx]["thetas"][0]).clone().detach(), getattr(model, model._arch_names[model.arch_idx]["thetas"][1]).clone().detach(), getattr(model, model._arch_names[model.arch_idx]["thetas"][2]).clone().detach()],
        num_classes=model._num_classes, layers=model._layers, Fch=model._Fch, width_mult_list=model._width_mult_list, stem_head_width=model._stem_head_width[0])

    plot_op(net.ops0, net.path0, F_base=args.Fch).savefig("table.png", bbox_inches="tight")
    logger.add_image("arch/ops0_arch%d"%model.arch_idx, np.swapaxes(np.swapaxes(plt.imread("table.png"), 0, 2), 1, 2), epoch)
    plot_op(net.ops1, net.path1, F_base=args.Fch).savefig("table.png", bbox_inches="tight")
    logger.add_image("arch/ops1_arch%d"%model.arch_idx, np.swapaxes(np.swapaxes(plt.imread("table.png"), 0, 2), 1, 2), epoch)
    plot_op(net.ops2, net.path2, F_base=args.Fch).savefig("table.png", bbox_inches="tight")
    logger.add_image("arch/ops2_arch%d"%model.arch_idx, np.swapaxes(np.swapaxes(plt.imread("table.png"), 0, 2), 1, 2), epoch)

    net.build_structure([2, 0])
    net = net.cuda()
    net.eval()
    latency0, _ = net.forward_latency(input_size[1:])
    logger.add_scalar("arch/fps0_arch%d"%model.arch_idx, 1000./latency0, epoch)
    logger.add_figure("arch/path_width_arch%d_02"%model.arch_idx, plot_path_width([2, 0], [net.path2, net.path0], [net.widths2, net.widths0]), epoch)

    net.build_structure([2, 1])
    net = net.cuda()
    net.eval()
    latency1, _ = net.forward_latency(input_size[1:])
    logger.add_scalar("arch/fps1_arch%d"%model.arch_idx, 1000./latency1, epoch)
    logger.add_figure("arch/path_width_arch%d_12"%model.arch_idx, plot_path_width([2, 1], [net.path2, net.path1], [net.widths2, net.widths1]), epoch)

    return 1000./latency0, 1000./latency1


if __name__ == '__main__':
    main(search=False,train=True)
