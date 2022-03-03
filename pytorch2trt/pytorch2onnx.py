from __future__ import division
import os
import sys
import time
import glob
import logging
from tqdm import tqdm
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
from tensorboardX import SummaryWriter

import numpy as np
from thop import profile
import onnx
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
from torch.autograd import Variable

##########seg########################
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import torch
import os
import time
from PIL import Image
import cv2
import torchvision
from config_train import config
import torchvision
import torch
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='cifar10 classification models')
parser.add_argument('--lr', default=0.1, help='')
parser.add_argument('--resume', default=None, help='')
parser.add_argument('--batch_size', type=int, default=768, help='')
parser.add_argument('--num_workers', type=int, default=4, help='')
parser.add_argument("--gpu_devices", type=int, nargs='+', default=None, help="")

parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:9318', type=str, help='')
parser.add_argument('--dist-backend', default='nccl', type=str, help='')
parser.add_argument('--rank', default=0, type=int, help='')
parser.add_argument('--world_size', default=1, type=int, help='')
parser.add_argument('--distributed', action='store_true', help='')
args = parser.parse_args()


args.gpu_devices = 0,
gpu_devices = ','.join([str(id) for id in args.gpu_devices])
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
config.gpu_devices = gpu_devices

###########################seg##################################
def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)

def get_img_np_nchw(filename):
    image = cv2.imread(filename)
    image_cv = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_cv = cv2.resize(image_cv, (224, 224))
    miu = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = np.array(image_cv, dtype=float) / 255.
    r = (img_np[:, :, 0] - miu[0]) / std[0]
    g = (img_np[:, :, 1] - miu[1]) / std[1]
    b = (img_np[:, :, 2] - miu[2]) / std[2]
    img_np_t = np.array([r, g, b])
    img_np_nchw = np.expand_dims(img_np_t, axis=0)
    return img_np_nchw

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        """Within this context, host_mom means the cpu memory and device means the GPU memory
        """
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream



def get_engine(max_batch_size=1, onnx_file_path="", engine_file_path="",\
               fp16_mode=False, int8_mode=False, save_engine=False,
              ):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine(max_batch_size, save_engine):
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        # create_network() without parameters will make parser.parse() return False
        with trt.Builder(TRT_LOGGER) as builder, \
                builder.create_network(flags=1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network,\
                trt.OnnxParser(network, TRT_LOGGER) as parser:
            
            builder.max_workspace_size = 1 << 30 # Your workspace size
            builder.max_batch_size = max_batch_size
            #pdb.set_trace()
            builder.fp16_mode = True  # Default: False
            builder.int8_mode = False  # Default: False
            if int8_mode:
                # To be updated
                raise NotImplementedError
                
            # Parse model file
            if not os.path.exists(onnx_file_path):
                quit('ONNX file {} not found'.format(onnx_file_path))
                
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                parser.parse(model.read())
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            #pdb.set_trace()
            #network.mark_output(network.get_layer(network.num_layers-1).get_output(0)) # Riz   
            #network.mark_output(network.get_layer(network.num_layers-1).get_output(1)) # Riz
            
            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")
            
            if save_engine:
                with open(engine_file_path, "wb") as f:
                    f.write(engine.serialize())
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, load it instead of building a new one.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine(max_batch_size, save_engine)


def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer data from CPU to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


def postprocess_the_outputs(h_outputs, shape_of_output):
    h_outputs = h_outputs.reshape(*shape_of_output)
    return h_outputs

def load_data(path):
    trans = T.Compose([
        T.Resize(1024,2048), 
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        
    ])
    
    img = Image.open(path)
    img_tensor = trans(img)
    return np.array(img_tensor)
def set_img_color(colors, background, img, gt, show255=False, weight_foreground=0.55):
    origin = np.array(img)
    for i in range(len(colors)):
        if i != background:
            img[np.where(gt == i)] = colors[i]
    if show255:
        img[np.where(gt == 255)] = 0
    cv2.addWeighted(img, weight_foreground, origin, (1 - weight_foreground), 0, img)
    return img
def show_prediction(colors, background, img, pred, weight_foreground=1):
    im = np.array(img, np.uint8)
    set_img_color(colors, background, im, pred, weight_foreground=weight_foreground)
    final = np.array(im)
    return final
def normalize(img, mean, std):
    # pytorch pretrained model need the input range: 0-1
    img = img.astype(np.float32) / 255.0
    img = img - mean
    img = img / std

    return img

def get_2dshape(shape, *, zero=True):
    if not isinstance(shape, collections.Iterable):
        shape = int(shape)
        shape = (shape, shape)
    else:
        h, w = map(int, shape)
        shape = (h, w)
    if zero:
        minv = 0
    else:
        minv = 1

    assert min(shape) >= minv, 'invalid shape: {}'.format(shape)
    return shape
def pad_image_to_shape(img, shape, border_mode, value):
    margin = np.zeros(4, np.uint32)
    shape = get_2dshape(shape)
    pad_height = shape[0] - img.shape[0] if shape[0] - img.shape[0] > 0 else 0
    pad_width = shape[1] - img.shape[1] if shape[1] - img.shape[1] > 0 else 0

    margin[0] = pad_height // 2
    margin[1] = pad_height // 2 + pad_height % 2
    margin[2] = pad_width // 2
    margin[3] = pad_width // 2 + pad_width % 2

    img = cv2.copyMakeBorder(img, margin[0], margin[1], margin[2], margin[3],
                             border_mode, value=value)

    return img, margin

def process_image(img, crop_size=None):
    p_img = img

    if img.shape[2] < 3:
        im_b = p_img
        im_g = p_img
        im_r = p_img
        p_img = np.concatenate((im_b, im_g, im_r), axis=2)

    p_img = normalize(p_img, image_mean, image_std)

    if crop_size is not None:
        p_img, margin = pad_image_to_shape(p_img, crop_size, cv2.BORDER_CONSTANT, value=0)
        p_img = p_img.transpose(2, 0, 1)

        return p_img, margin

    p_img = p_img.transpose(2, 0, 1)

    return p_img





cityscapes_trainID2id = {
  0: 7,
  1: 8,
  2: 11,
  3: 12,
  4: 13,
  5: 17,
  6: 19,
  7: 20,
  8: 21,
  9: 22,
  10: 23,
  11: 24,
  12: 25,
  13: 26,
  14: 27,
  15: 28,
  16: 31,
  17: 32,
  18: 33,
  19: 0
}



################################################################


def adjust_learning_rate(base_lr, power, optimizer, epoch, total_epoch):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * power

def main():
  

    args = parser.parse_args()
    ngpus_per_node = torch.cuda.device_count()

    logger = SummaryWriter(config.save)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(config.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
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
    distill_criterion = nn.KLDivLoss().cuda()

    # data loader ###########################
    if config.is_test:
        data_setting = {'img_root': config.img_root_folder,
                        'gt_root': config.gt_root_folder,
                        'train_source': config.train_source,
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

    train_loader = get_train_loader(config, Cityscapes, test=config.is_test)


    # Model #######################################
    models = []
    evaluators = []
    testers = []
    lasts = []
    idx = 1
    arch_idx = 1
    state = torch.load(os.path.join(config.load_path, "arch_%d_%d.pt"%(arch_idx, int(config.load_epoch))))
    model = Network(
            [state["alpha_%d_0"%arch_idx].detach(), state["alpha_%d_1"%arch_idx].detach(), state["alpha_%d_2"%arch_idx].detach()],
            [None, state["beta_%d_1"%arch_idx].detach(), state["beta_%d_2"%arch_idx].detach()],
            [state["ratio_%d_0"%arch_idx].detach(), state["ratio_%d_1"%arch_idx].detach(), state["ratio_%d_2"%arch_idx].detach()],
            num_classes=config.num_classes, layers=config.layers, Fch=config.Fch, width_mult_list=config.width_mult_list, stem_head_width=config.stem_head_width[idx])
    model.cuda()
   
    mIoU02 = state["mIoU02"]; latency02 = state["latency02"]; obj02 = objective_acc_lat(mIoU02, latency02)
    mIoU12 = state["mIoU12"]; latency12 = state["latency12"]; obj12 = objective_acc_lat(mIoU12, latency12)
    if obj02 > obj12: last = [2, 0]
    else: last = [2, 1]
    lasts.append(last)
    model.build_structure(last)
    logging.info("net: " + str(model))
    for b in last:
        if len(config.width_mult_list) > 1:
           plot_op(getattr(model, "ops%d"%b), getattr(model, "path%d"%b), width=getattr(model, "widths%d"%b), head_width=config.stem_head_width[idx][1], F_base=config.Fch).savefig(os.path.join(config.save, "ops_%d_%d.png"%(arch_idx,b)), bbox_inches="tight")
        else:
           plot_op(getattr(model, "ops%d"%b), getattr(model, "path%d"%b), F_base=config.Fch).savefig(os.path.join(config.save, "ops_%d_%d.png"%(arch_idx,b)), bbox_inches="tight")
    plot_path_width(model.lasts, model.paths, model.widths).savefig(os.path.join(config.save, "path_width%d.png"%arch_idx))
    plot_path_width([2, 1, 0], [model.path2, model.path1, model.path0], [model.widths2, model.widths1, model.widths0]).savefig(os.path.join(config.save, "path_width_all%d.png"%arch_idx))
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('The number of parameters of model is', num_params)
       
    model = model.cuda()
    init_weight(model, nn.init.kaiming_normal_, torch.nn.BatchNorm2d, config.bn_eps, config.bn_momentum, mode='fan_in', nonlinearity='relu')
    if arch_idx == 0 and len(config.arch_idx) > 1:
       model = smp.DeepLabV3Plus(encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
			encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
			in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
			classes=19,                      # model output channels (number of classes in your dataset)
		).cuda()
    elif config.is_eval:
         partial = torch.load(os.path.join(config.eval_path, "weights%d.pt"%arch_idx))
         new_state_dict = OrderedDict()
         for k, v in partial.items():
             name = k[7:] # module字段在最前面，从第7个字符开始就可以去掉module
             new_state_dict[name] = v #新字典的key值对应的value一一对应
         state = model.state_dict()
         pretrained_dict = {k: v for k, v in new_state_dict.items() if k in state}
         state.update(pretrained_dict)
         model.load_state_dict(state)

    evaluator = SegEvaluator(Cityscapes(data_setting, 'val', None), config.num_classes, config.image_mean,
                                 config.image_std, model, config.eval_scale_array, config.eval_flip, 0, out_idx=0, config=config,
                                 verbose=False, save_path=None, show_image=False, show_prediction=False)
    evaluators.append(evaluator)
    tester = SegTester(Cityscapes(data_setting, 'test', None), config.num_classes, config.image_mean,
                                 config.image_std, model, config.eval_scale_array, config.eval_flip, 0, out_idx=0, config=config,
                                 verbose=False, save_path=None, show_prediction=False)
    testers.append(tester)

        # Optimizer ###################################
    base_lr = config.lr
    if arch_idx == 1 or len(config.arch_idx) == 1:
       optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=config.momentum, weight_decay=config.weight_decay)
       models.append(model)


    # Cityscapes ###########################################
    #if config.is_eval:
    #    logging.info(config.load_path)
    #    logging.info(config.eval_path)
    #    logging.info(config.save)
    #    with torch.no_grad():
    #        if config.is_test:
                # test
    #            print("[test...]")
    #            with torch.no_grad():
    #                test(0, models, testers, logger)
    input_name = ['input']
    output_name = ['output']
    input = Variable(torch.randn(1, 3, 512, 1024)).cuda()#固定计算图
    torch.onnx.export(model, input, 'seg21_512.onnx', input_names=input_name, output_names=output_name, verbose=True,opset_version=11)
    test = onnx.load('seg21.onnx')
    onnx.checker.check_model(test)
    print("==> Passed")     
    
    

def train(train_loader, models, criterion, distill_criterion, optimizer, logger, epoch):
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
                target = target.cuda(non_blocking=True)
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
            description += "[mIoU%d: %.3f]"%(arch_idx, metrics[idx].get_scores())

        pbar.set_description("[Step %d/%d]"%(step + 1, len(train_loader)) + description)
        logger.add_scalar('loss/train', loss+loss_kl, epoch*len(pbar)+step)

        loss.backward()
        optimizer.step()

    return [ metric.get_scores() for metric in metrics ]


def infer(models, evaluators, logger):
    mIoUs = []
    for model, evaluator in zip(models, evaluators):
        model.eval()
        _, mIoU = evaluator.run_online()
        #_, mIoU = evaluator.run_online_multiprocess()
        mIoUs.append(mIoU)
    return mIoUs

def test(epoch, models, testers, logger):
    for idx, arch_idx in enumerate(config.arch_idx):
        if arch_idx == 0: continue
        model = models[idx]
        tester = testers[idx]
        os.system("mkdir %s"%os.path.join(os.path.join(os.path.realpath('.'), config.save, "test")))
        model.eval()
        tester.run_online()
        os.system("mv %s %s"%(os.path.join(os.path.realpath('.'), config.save, "test"), os.path.join(os.path.realpath('.'), config.save, "test_%d_%d"%(arch_idx, epoch))))


if __name__ == '__main__':
    main() 
