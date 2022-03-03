# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import sys
import numpy as np
from easydict import EasyDict as edict
import time
C = edict()
config = C
cfg = C

C.seed = 12345

"""please config ROOT_dir and user when u first using"""
C.repo_name = 'AutoSeg_edge'
C.abs_dir = osp.realpath(".")
C.this_dir = C.abs_dir.split(osp.sep)[-1]
C.root_dir = C.abs_dir[:C.abs_dir.index(C.repo_name) + len(C.repo_name)]

"""Data Dir"""
C.dataset_path = "/home/wangshuo/douzi/pytorch-multigpu/cityscapes"
C.img_root_folder = C.dataset_path
C.gt_root_folder = C.dataset_path
C.train_source = osp.join(C.dataset_path, "cityscapes_train_fine.txt")
C.eval_source = osp.join(C.dataset_path, "cityscapes_val_fine.txt")
C.test_source = osp.join(C.dataset_path, "cityscapes_test.txt")

"""Path Config"""
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

add_path(osp.join(C.root_dir, 'tools'))

"""Image Config"""
C.num_classes = 19
C.background = -1
C.image_mean = np.array([0.485, 0.456, 0.406])
C.image_std = np.array([0.229, 0.224, 0.225])
C.down_sampling = 2 # use downsampled images during search. In dataloader the image will first be down_sampled then cropped
C.image_height = 160 # crop height after down_sampling in dataloader
C.image_width = 160*2 # crop width after down_sampling in dataloader
C.image_size = C.image_height * C.image_width
C.gt_down_sampling = 8 # model's default output size without final upsampling
C.num_train_imgs = 2975 # number of training images
C.num_eval_imgs = 20 # number of validation images

"""dist"""
C.gpu = 0
C.gpu_devices =0
C.dist_url ='tcp://127.0.0.1:3456'
C.dist_backend ='nccl'
C.rank = 0
C.world_size =1

""" Settings for network, this would be different for each kind of model"""
C.bn_eps = 1e-5 # a value added to the BN denominator for numerical stability
C.bn_momentum = 0.1 # value used for the running_mean and running_var computation

"""Train Config"""
C.lr = 0.01 # learning rate for updating supernet weight (NOT arch params)
C.momentum = 0.9 # SGD momentum
C.weight_decay = 5e-4 # SGD weight decay
C.num_workers = 4 # workers for dataloader
C.train_scale_array = [0.75, 1, 1.25] # scale factors for augmentation during training

"""Eval Config"""
C.eval_stride_rate = 5 / 6 # stride for crop based evaluation. Not used in this repo
C.eval_scale_array = [1, ]# multi-scale evaluation
C.eval_flip = False # flipping for evaluation
C.eval_height = 1024 # real image height
C.eval_width = 2048 # real image width


""" Search Config """
C.grad_clip = 5 # grad clip for search
C.train_portion = 0.5 # use how much % of training data for search
C.arch_learning_rate = 3e-4 # learning rate for updating arch params
C.arch_weight_decay = 0
C.layers = 10 # layers (cells) for supernet
C.branch = 2 # number of output branches

#C.pretrain = True
C.pretrain = "search-224x448_F12.L16_batch2-20210728-155926 "
########################################
C.prun_modes = ['max', 'arch_ratio',] # channel pruning mode for [teacher, student], i.e. by default teacher will use max channel number, and student will sample the channel number based on arch_ratio
C.Fch = 6 # base channel number
C.width_mult_list = [4./12, 6./12, 8./12, 10./12, 1.] # selection choices for channel pruning
C.stem_head_width = [(1, 1), (8./12, 8./12),] # width ratio (#channel / Fch) for [teacher, student]
C.FPS_min = [0, 100.] # minimum FPS required for [teacher, student]
C.FPS_max = [0, 150.] # maximum FPS allowed for [teacher, student]
if C.pretrain == True:
    C.batch_size = 4
    C.niters_per_epoch = max(C.num_train_imgs // 2 // C.batch_size, 400)
    C.lr = 2e-2
    C.latency_weight = [0, 0] # weight of latency penalty loss
    C.image_height = 256 # this size is after down_sampling
    C.image_width = 256*2
    C.nepochs = 20
    C.save = "pretrain-%dx%d_F%d.L%d_batch%d"%(C.image_height, C.image_width, C.Fch, C.layers, C.batch_size)
else:
    C.batch_size = 12
    C.niters_per_epoch = max(C.num_train_imgs // 2 // C.batch_size, 400)
    C.latency_weight = [0, 1e-2,]
    C.image_height = 224 # this size is after down_sampling
    C.image_width = 224*2
    C.nepochs = 30
    C.save = "%dx%d_F%d.L%d_batch%d"%(C.image_height, C.image_width, C.Fch, C.layers, C.batch_size)
########################################
assert len(C.latency_weight) == len(C.stem_head_width) and len(C.stem_head_width) == len(C.FPS_min) and len(C.FPS_min) == len(C.FPS_max)
C.slimmable = True

C.train_portion = 0.5

C.unrolled = False

C.alpha_weight = 1/4
C.ratio_weight = 2/4
C.beta_weight = 1/4
C.flops_weight = [0, 1e-9,]


C.Lantency = 1
C.Flops = 1

C.flops_max = 6.1e8
C.flops_min = 5.5e8



C.log_latency = 0
C.log_Flops = 0
C.Flops_target = 1.4
C.Latency_target = 0.5

C.Latency_precision = 0
C.Flops_precision = 0
C.Segm_precision = 0


# C.loss_weight = [1e-2, 0, 1, 0]


C.unrolled = False # for DARTS v2
