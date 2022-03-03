__all__ = ['ConvNorm', 'Conv3x3', 'BasicResidual2x',  "DwsBlock", "DWConv", "FusedBlock", 'OPS', 'OPS_name', 'OPS_Class']

from functools import partial
from pdb import set_trace as bp
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from thop import profile
import sys
import os.path as osp
from easydict import EasyDict as edict

from config_search import config
C = edict()
"""please config ROOT_dir and user when u first using"""
C.repo_name = 'FasterSeg'
C.abs_dir = osp.realpath(".")
C.this_dir = C.abs_dir.split(osp.sep)[-1]
C.root_dir = C.abs_dir[:C.abs_dir.index(C.repo_name) + len(C.repo_name)]
"""Path Config"""
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

add_path(osp.join(C.root_dir, 'tools'))
try:
    from utils.darts_utils import compute_latency_ms_tensorrt as compute_latency
    print("use TensorRT for latency test")
except:
    from utils.darts_utils import compute_latency_ms_pytorch as compute_latency
    print("use PyTorch for latency test")
from slimmable_ops import USConv2d, USBatchNorm2d


latency_lookup_table = {}
table_file_name = "latency_lookup_table.npy"
if osp.isfile(table_file_name):
    latency_lookup_table = np.load(table_file_name, allow_pickle=True).item()

flops_lookup_table = {}
table_file_name = "flops_lookup_table.npy"
if osp.isfile(table_file_name):
    flops_lookup_table = np.load(table_file_name, allow_pickle=True).item()

BatchNorm2d = nn.BatchNorm2d



def get_same_padding_conv2d(image_size=None):
    """Chooses static padding if you have specified an image size, and dynamic padding otherwise.
       Static padding is necessary for ONNX exporting of models.

    Args:
        image_size (int or tuple): Size of the image.

    Returns:
        Conv2dDynamicSamePadding or Conv2dStaticSamePadding.
    """
    if image_size is None:
        return Conv2dDynamicSamePadding
    else:
        return partial(Conv2dStaticSamePadding, image_size=image_size)


class Conv2dStaticSamePadding(nn.Conv2d):
    """2D Convolutions like TensorFlow's 'SAME' mode, with the given input image size.
       The padding mudule is calculated in construction function, then used in forward.
    """

    # With the same calculation as Conv2dDynamicSamePadding

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, image_size=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, stride, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

        # Calculate padding based on image size and save it
        assert image_size is not None
        ih, iw = (image_size, image_size) if isinstance(image_size, int) else image_size
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2,
                                                pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = nn.Identity()

    def set_ratio(self, ratio):
        self.ratio = ratio

    def forward(self, x):
        x = self.static_padding(x)
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x



class Conv2dDynamicSamePadding(nn.Conv2d):
    """2D Convolutions like TensorFlow, for a dynamic image size.
       The padding is operated in forward function by calculating dynamically.
    """

    # Tips for 'SAME' mode padding.
    #     Given the following:
    #         i: width or height
    #         s: stride
    #         k: kernel size
    #         d: dilation
    #         p: padding
    #     Output after Conv2d:
    #         o = floor((i+p-((k-1)*d+1))/s+1)
    # If o equals i, i = floor((i+p-((k-1)*d+1))/s+1),
    # => p = (i-1)*s+((k-1)*d+1)-i

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

    def set_ratio(self, ratio):
        self.ratio = ratio

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)  # change the output size according to stride ! ! !
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

# A memory-efficient implementation of Swish function
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class ConvNorm(nn.Module):
    def __init__(self, C_in, C_out, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False,
                 slimmable=True, width_mult_list=[1.]):
        super(ConvNorm, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        assert stride in [1, 2]
        self.stride = stride
        if padding is None:
            # assume h_out = h_in / s
            self.padding = int(np.ceil((dilation * (kernel_size - 1) + 1 - stride) / 2.))
        else:
            self.padding = padding
        self.dilation = dilation
        assert type(groups) == int
        if kernel_size == 1:
            self.groups = 1
        else:
            self.groups = groups
        self.bias = bias
        self.slimmable = slimmable
        self.width_mult_list = width_mult_list
        self.ratio = (1., 1.)

        
        self.conv = USConv2d(C_in, C_out, kernel_size, stride, padding=self.padding, dilation=dilation, groups=self.groups, bias=bias, width_mult_list=width_mult_list)
        self.bn = USBatchNorm2d(C_out, width_mult_list)
        self.relu = nn.ReLU(inplace=True)

      
    def set_ratio(self, ratio):
        assert self.slimmable
        assert len(ratio) == 2
        self.ratio = ratio
        self.conv.set_ratio(ratio)
        self.bn.set_ratio(ratio[1])

    @staticmethod
    def _flops(h, w, C_in, C_out, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False):
        layer = ConvNorm(C_in, C_out, kernel_size, stride, padding, dilation, groups, bias, slimmable=False)
        flops, params = profile(layer, inputs=(torch.randn(1, C_in, h, w),))
        return flops

    @staticmethod
    def _latency(h, w, C_in, C_out, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False):
        layer = ConvNorm(C_in, C_out, kernel_size, stride, padding, dilation, groups, bias, slimmable=False)
        latency = compute_latency(layer, (1, C_in, h, w))
        return latency

    def forward_latency(self, size):
        c_in, h_in, w_in = size
        if self.slimmable:
            assert c_in == make_divisible(self.C_in * self.ratio[0]), "c_in %d, self.C_in * self.ratio[0] %d" % (
            c_in, self.C_in * self.ratio[0])
            c_out = make_divisible(self.C_out * self.ratio[1])
        else:
            assert c_in == self.C_in, "c_in %d, self.C_in %d" % (c_in, self.C_in)
            c_out = self.C_out
        if self.stride == 1:
            h_out = h_in;
            w_out = w_in
        else:
            h_out = h_in // 2;
            w_out = w_in // 2
        name = "ConvNorm_H%d_W%d_Cin%d_Cout%d_kernel%d_stride%d" % (
        h_in, w_in, c_in, c_out, self.kernel_size, self.stride)
        if name in latency_lookup_table:
            latency = latency_lookup_table[name]
        else:
            print("not found in latency_lookup_table:", name)
            latency = ConvNorm._latency(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.padding,
                                        self.dilation, self.groups, self.bias)
            latency_lookup_table[name] = latency
            np.save(table_file_name, latency_lookup_table)
        return latency, (c_out, h_out, w_out)

    def forward_flops(self, size):
        c_in, h_in, w_in = size
        if self.slimmable:
            assert c_in == make_divisible(self.C_in * self.ratio[0]), "c_in %d, self.C_in * self.ratio[0] %d" % (
            c_in, self.C_in * self.ratio[0])
            c_out = make_divisible(self.C_out * self.ratio[1])
        else:
            assert c_in == self.C_in, "c_in %d, self.C_in %d" % (c_in, self.C_in)
            c_out = self.C_out
        if self.stride == 1:
            h_out = h_in;
            w_out = w_in
        else:
            h_out = h_in // 2;
            w_out = w_in // 2
        name = "ConvNorm_H%d_W%d_Cin%d_Cout%d_kernel%d_stride%d" % (
        h_in, w_in, c_in, c_out, self.kernel_size, self.stride)
        if name in flops_lookup_table:
            flops = flops_lookup_table[name]
        else:
            print("not found in flops_lookup_table:", name)
            flops = ConvNorm._flops(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.padding, self.dilation,
                                    self.groups, self.bias)
            flops_lookup_table[name] = flops
            np.save(table_file_name, flops_lookup_table)
        # if quantize:
        #     flops /= 4

        return flops, (c_out, h_out, w_out)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv3x3(nn.Module):
    def __init__(self, C_in, C_out, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False, slimmable=False, width_mult_list=[1.]):
        super(Conv3x3, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        if kernel_size == 1:
            self.groups = 1
        else:
            self.groups = groups

        self.slimmable = slimmable
        self.width_mult_list = width_mult_list
        assert stride in [1, 2]
        if self.stride == 2: self.dilation = 1
        self.ratio = (1., 1.)
        if padding is None:
            # assume h_out = h_in / s
            self.padding = int(np.ceil((dilation * (kernel_size - 1) + 1 - stride) / 2.))
        else:
            self.padding = padding

        self.relu = nn.ReLU(inplace=True)

        self.conv = USConv2d(C_in, C_out, kernel_size, stride, padding=self.padding, dilation=dilation, groups=self.groups, bias=bias, width_mult_list=width_mult_list)
        self.bn = USBatchNorm2d(C_out, width_mult_list)
        self.relu = nn.ReLU(inplace=True)

        
    def set_ratio(self, ratio):
        assert self.slimmable
        assert len(ratio) == 2
        self.ratio = ratio
        self.conv.set_ratio(ratio)
        self.bn.set_ratio(ratio[1])

    @staticmethod
    def _flops(h, w, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1):
        layer = Conv3x3(C_in, C_out, kernel_size, stride, dilation, groups, slimmable=False)
        flops, params = profile(layer, inputs=(torch.randn(1, C_in, h, w),))
        return flops

    @staticmethod
    def _latency(h, w, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1):
        layer = Conv3x3(C_in, C_out, kernel_size, stride, dilation, groups, slimmable=False)
        latency = compute_latency(layer, (1, C_in, h, w))
        return latency

    def forward_latency(self, size):
        c_in, h_in, w_in = size
        if self.slimmable:
            assert c_in == make_divisible(self.C_in * self.ratio[0]), "c_in %d, int(self.C_in * self.ratio[0]) %d" % (
            c_in, int(self.C_in * self.ratio[0]))
            c_out = make_divisible(self.C_out * self.ratio[1])
        else:
            assert c_in == self.C_in, "c_in %d, self.C_in %d" % (c_in, self.C_in)
            c_out = self.C_out
        if self.stride == 1:
            h_out = h_in;
            w_out = w_in
        else:
            h_out = h_in // 2;
            w_out = w_in // 2
        name = "Conv3x3_H%d_W%d_Cin%d_Cout%d_stride%d_dilation%d" % (
        h_in, w_in, c_in, c_out, self.stride, self.dilation)
        if name in latency_lookup_table:
            latency = latency_lookup_table[name]
        else:
            print("not found in latency_lookup_table:", name)
            latency = Conv3x3._latency(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.dilation,
                                       self.groups)
            latency_lookup_table[name] = latency
            np.save(table_file_name, latency_lookup_table)
        return latency, (c_out, h_out, w_out)

    def forward_flops(self, size, quantize=False):
        c_in, h_in, w_in = size
        if self.slimmable:
            assert c_in == make_divisible(self.C_in * self.ratio[0]), "c_in %d, self.C_in * self.ratio[0] %d" % (
            c_in, self.C_in * self.ratio[0])
            c_out = make_divisible(self.C_out * self.ratio[1])
        else:
            assert c_in == self.C_in, "c_in %d, self.C_in %d" % (c_in, self.C_in)
            c_out = self.C_out
        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2
        name = "Conv3x3_H%d_W%d_Cin%d_Cout%d_kernel%d_stride%d" % ( h_in, w_in, c_in, c_out, self.kernel_size, self.stride)
        if name in flops_lookup_table:
            flops = flops_lookup_table[name]
        else:
            print("not found in flops_lookup_table:", name)
            flops = Conv3x3._flops(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.dilation, self.groups)
            flops_lookup_table[name] = flops
            np.save(table_file_name, flops_lookup_table)
        # if quantize:
        #     flops /= 4

        return flops, (c_out, h_out, w_out)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class BasicResidual2x(nn.Module):
    def __init__(self, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1, slimmable=True,
                 width_mult_list=[1.]):
        super(BasicResidual2x, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.slimmable = slimmable
        self.width_mult_list = width_mult_list
        assert stride in [1, 2]
        if self.stride == 2: self.dilation = 1
        self.ratio = (1., 1.)

        self.relu = nn.ReLU(inplace=True)
    
    
        self.conv1 = USConv2d(C_in, C_out, 3, stride, padding=dilation, dilation=dilation, groups=groups,
                                  bias=False, width_mult_list=width_mult_list)
        self.bn1 = USBatchNorm2d(C_out, width_mult_list)
        self.conv2 = USConv2d(C_out, C_out, 3, 1, padding=dilation, dilation=dilation, groups=groups, bias=False,
                                  width_mult_list=width_mult_list)
        self.bn2 = USBatchNorm2d(C_out, width_mult_list)
        

    def set_ratio(self, ratio):
        assert len(ratio) == 2
        self.ratio = ratio
        self.conv1.set_ratio(ratio)
        self.bn1.set_ratio(ratio[1])
        self.conv2.set_ratio((ratio[1], ratio[1]))
        self.bn2.set_ratio(ratio[1])

    @staticmethod
    def _flops(h, w, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1):
        layer = BasicResidual2x(C_in, C_out, kernel_size, stride, dilation, groups, slimmable=False)
        flops, params = profile(layer, inputs=(torch.randn(1, C_in, h, w),), verbose=False)
        return flops

    @staticmethod
    def _latency(h, w, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1):
        layer = BasicResidual2x(C_in, C_out, kernel_size, stride, dilation, groups, slimmable=False)
        latency = compute_latency(layer, (1, C_in, h, w))
        return latency

    def forward_latency(self, size):
        c_in, h_in, w_in = size
        if self.slimmable:
            assert c_in == int(self.C_in * self.ratio[0])
            c_out = int(self.C_out * self.ratio[1])
        else:
            assert c_in == self.C_in, "c_in %d, self.C_in%d" % (c_in, self.C_in)
            c_out = self.C_out
        if self.stride == 1:
            h_out = h_in;
            w_out = w_in
        else:
            h_out = h_in // 2;
            w_out = w_in // 2
        name = "BasicResidual2x_H%d_W%d_Cin%d_Cout%d_stride%d_dilation%d" % (
        h_in, w_in, c_in, c_out, self.stride, self.dilation)
        if name in latency_lookup_table:
            latency = latency_lookup_table[name]
        else:
            print("not found in flops_lookup_table:", name)
            latency = BasicResidual2x._latency(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.dilation,
                                               self.groups)
            latency_lookup_table[name] = latency
            np.save(table_file_name, latency_lookup_table)
        return latency, (c_out, h_out, w_out)

    def forward_flops(self, size):
        c_in, h_in, w_in = size
        if self.slimmable:
            assert c_in == make_divisible(self.C_in * self.ratio[0]), "c_in %d, self.C_in * self.ratio[0] %d" % (
            c_in, self.C_in * self.ratio[0])
            c_out = make_divisible(self.C_out * self.ratio[1])
        else:
            assert c_in == self.C_in, "c_in %d, self.C_in %d" % (c_in, self.C_in)
            c_out = self.C_out
        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2
        name = "BasicResidual2x_H%d_W%d_Cin%d_Cout%d_kernel%d_stride%d" % ( h_in, w_in, c_in, c_out, self.kernel_size, self.stride)
        if name in flops_lookup_table:
            flops = flops_lookup_table[name]
        else:
            print("not found in flops_lookup_table:", name)
            flops = BasicResidual2x._flops(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.dilation, self.groups)
            flops_lookup_table[name] = flops
            np.save(table_file_name, flops_lookup_table)
        # if quantize:
        #     flops /= 4

        return flops, (c_out, h_out, w_out)
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out

from collections import OrderedDict

def make_divisible(v, divisor=8, min_value=1):
    """
    forked from slim:
    https://github.com/tensorflow/models/blob/\
    0344c5503ee55e24f0de7f37336a6e08f10976fd/\
    research/slim/nets/mobilenet/mobilenet.py#L62-L69
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class DWConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1, slimmable=True,
                 width_mult_list=[1.]):
        super(DWConv, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.slimmable = slimmable
        self.width_mult_list = width_mult_list
        assert stride in [1, 2]
        if self.stride == 2: self.dilation = 1
        self.ratio = (1., 1.)

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = USConv2d(C_in, C_in , 1, 1, padding=0, dilation=dilation, groups=groups, bias=False, width_mult_list=width_mult_list)
        self.bn1 = USBatchNorm2d(C_in, width_mult_list)

        self.conv2 = USConv2d(C_in, C_in, 3, stride, padding=dilation, dilation=dilation, groups=C_in, bias=False, width_mult_list=width_mult_list)
        self.bn2 = USBatchNorm2d(C_in , width_mult_list)

        self.conv3 = USConv2d(C_in , C_out, 1, 1, padding=0, dilation=dilation, groups=groups, bias=False,  width_mult_list=width_mult_list)
        self.bn3 = USBatchNorm2d(C_out, width_mult_list)
       

    def set_ratio(self, ratio):
        assert len(ratio) == 2
        self.ratio = ratio
        self.conv1.set_ratio((ratio[0], 1))
        self.bn1.set_ratio(1)
        self.conv2.set_ratio((1, 1))
        self.bn2.set_ratio(1)
        self.conv3.set_ratio((1, ratio[1]))
        self.bn3.set_ratio(ratio[1])


    @staticmethod
    def _flops(h, w, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1):
        layer = DWConv(C_in, C_out, kernel_size, stride, dilation, groups, slimmable=False)
        flops, params = profile(layer, inputs=(torch.randn(1, C_in, h, w),), verbose=False)
        return flops

    @staticmethod
    def _latency(h, w, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1):
        layer = DWConv(C_in, C_out, kernel_size, stride, dilation, groups, slimmable=False)
        latency = compute_latency(layer, (1, C_in, h, w))
        return latency

    def forward_latency(self, size):
        c_in, h_in, w_in = size
        if self.slimmable:
            assert c_in == int(self.C_in * self.ratio[0])
            c_out = int(self.C_out * self.ratio[1])
        else:
            assert c_in == self.C_in, "c_in %d, self.C_in%d" % (c_in, self.C_in)
            c_out = self.C_out
        if self.stride == 1:
            h_out = h_in;
            w_out = w_in
        else:
            h_out = h_in // 2;
            w_out = w_in // 2
        name = "DWConv_H%d_W%d_Cin%d_Cout%d_stride%d_dilation%d" % (
        h_in, w_in, c_in, c_out, self.stride, self.dilation)
        if name in latency_lookup_table:
            latency = latency_lookup_table[name]
        else:
            print("not found in latency_lookup_table:", name)
            latency = DWConv._latency(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.dilation,
                                               self.groups)
            latency_lookup_table[name] = latency
            np.save(table_file_name, latency_lookup_table)
        return latency, (c_out, h_out, w_out)

    def forward_flops(self, size):
        c_in, h_in, w_in = size
        if self.slimmable:
            assert c_in == make_divisible(self.C_in * self.ratio[0]), "c_in %d, self.C_in * self.ratio[0] %d" % (
                c_in, self.C_in * self.ratio[0])
            c_out = make_divisible(self.C_out * self.ratio[1])
        else:
            assert c_in == self.C_in, "c_in %d, self.C_in %d" % (c_in, self.C_in)
            c_out = self.C_out
        if self.stride == 1:
            h_out = h_in;
            w_out = w_in
        else:
            h_out = h_in // 2;
            w_out = w_in // 2
        name = "DWConv_H%d_W%d_Cin%d_Cout%d_kernel%d_stride%d" % (
        h_in, w_in, c_in, c_out, self.kernel_size, self.stride)
        if name in flops_lookup_table:
            flops = flops_lookup_table[name]
        else:
            print("not found in flops_lookup_table:", name)
            flops = DWConv._flops(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.dilation,
                                           self.groups)
            flops_lookup_table[name] = flops
            np.save(table_file_name, flops_lookup_table)
        # if quantize:
        #     flops /= 4

        return flops, (c_out, h_out, w_out)
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        return out

class DwsBlock(nn.Module):
    def __init__(self, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1, slimmable=True,
                 width_mult_list=[1.]):
        super(DwsBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.slimmable = slimmable
        self.width_mult_list = width_mult_list
        assert stride in [1, 2]
        if self.stride == 2: self.dilation = 1
        self.ratio = (1., 1.)

        self.relu = nn.ReLU(inplace=True)

       
        self.conv1 = USConv2d(C_in, C_in*4 , 1, 1, padding=0, dilation=dilation, groups=groups, bias=False, width_mult_list=width_mult_list)
        self.bn1 = USBatchNorm2d(int(C_in*4), width_mult_list)

        self.conv2 = USConv2d(C_in*4, C_in*4, 3, stride, padding=dilation, dilation=dilation, groups=C_in*4, bias=False, width_mult_list=width_mult_list)
        self.bn2 = USBatchNorm2d(C_in*4 , width_mult_list)

        self.conv3 = USConv2d(C_in*4 , C_out, 1, 1, padding=0, dilation=dilation, groups=groups, bias=False,  width_mult_list=width_mult_list)
        self.bn3 = USBatchNorm2d(C_out, width_mult_list)

        self.skip = USConv2d(C_in, C_out, 1, stride, padding=0, dilation=dilation, groups=1, bias=False, width_mult_list=width_mult_list)
        self.bn4 = USBatchNorm2d(C_out, width_mult_list)

       
    def set_ratio(self, ratio):
        assert len(ratio) == 2
        self.ratio = ratio
        self.conv1.set_ratio((ratio[0], 1))
        self.bn1.set_ratio(1)
        self.conv2.set_ratio((1, 1))
        self.bn2.set_ratio(1)
        self.conv3.set_ratio((1, ratio[1]))
        self.bn3.set_ratio(ratio[1])

        if hasattr(self, 'skip'):
            self.skip.set_ratio(ratio)
            self.bn4.set_ratio(ratio[1])

    @staticmethod
    def _flops(h, w, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1):
        layer = MBCBlock(C_in, C_out, kernel_size, stride, dilation, groups=1, slimmable=False)
        flops, params = profile(layer, inputs=(torch.randn(1, C_in, h, w),))
        return flops

    @staticmethod
    def _latency(h, w, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1):
        layer = MBCBlock(C_in, C_out, kernel_size, stride, dilation, groups, slimmable=False)
        latency = compute_latency(layer, (1, C_in, h, w))
        return latency

    def forward_latency(self, size):
        c_in, h_in, w_in = size
        if self.slimmable:
            c_out = make_divisible(self.C_out * self.ratio[1])
        else:
            c_out = self.C_out
        if self.stride == 1:
            h_out = h_in;
            w_out = w_in
        else:
            h_out = h_in // 2;
            w_out = w_in // 2
        name = "DwsBlock_H%d_W%d_Cin%d_Cout%d_stride%d_dilation%d" % (
        h_in, w_in, c_in, c_out, self.stride, self.dilation)
        if name in latency_lookup_table:
            latency = latency_lookup_table[name]
        else:
            print("not found in latency_lookup_table:", name)
            latency = DwsBlock._latency(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.dilation,
                                        self.groups)
            latency_lookup_table[name] = latency
            np.save(table_file_name, latency_lookup_table)
        return latency, (c_out, h_out, w_out)

    def forward_flops(self, size, quantize=False):
        c_in, h_in, w_in = size
        if self.slimmable:
            c_out = make_divisible(self.C_out * self.ratio[1])
        else:
            c_out = self.C_out
        if self.stride == 1:
            h_out = h_in;
            w_out = w_in
        else:
            h_out = h_in // 2;
            w_out = w_in // 2
        name = "DwsBlock_H%d_W%d_Cin%d_Cout%d_kernel%d_stride%d" % (
        h_in, w_in, c_in, c_out, self.kernel_size, self.stride)
        if name in flops_lookup_table:
            flops = flops_lookup_table[name]
        else:
            print("not found in flops_lookup_table:", name)
            flops = DwsBlock._flops(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.dilation, self.groups)
            flops_lookup_table[name] = flops
            np.save(table_file_name, flops_lookup_table)
        # if quantize:
        #     ratio_dws = 3*3 / (3*3 + self.C_out)
        #     flops = ratio_dws * flops + (1-ratio_dws) * flops / 4

        return flops, (c_out, h_out, w_out)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if hasattr(self, 'skip'):
            identity = self.bn4(self.skip(identity))

        out += identity
        out = self.relu(out)

        return out


class MBCBlock(nn.Module):
    def __init__(self, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1, slimmable=True,
                 width_mult_list=[1.]):
        super(MBCBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.slimmable = slimmable
        self.width_mult_list = width_mult_list
        assert stride in [1, 2]
        if self.stride == 2: self.dilation = 1
        self.ratio = (1., 1.)

        self.relu = nn.ReLU(inplace=True)

        
        self.conv1 = USConv2d(C_in, C_in * 4, 1, 1, padding=0, dilation=dilation, groups=1, bias=False,
                                  width_mult_list=width_mult_list)
        self.bn1 = USBatchNorm2d(int(C_in * 4), width_mult_list)

        self.conv2 = USConv2d(C_in * 4, C_in * 4, 3, stride, padding=dilation, dilation=dilation, groups=C_in * 4,  bias=False, width_mult_list=width_mult_list)
        self.bn2 = USBatchNorm2d(C_in * 4, width_mult_list)

        num_squeezed_channels = max(1, int(C_in * 0.25))
        self.squeeze = nn.AdaptiveAvgPool2d((1,1))
        self._se_reduce = nn.Conv2d(C_in * 4, num_squeezed_channels, 1, 1,0)
        self._se_expand = USConv2d(num_squeezed_channels, C_in * 4, 1, 1,0)
        self._swish = MemoryEfficientSwish()

        self.conv3 = USConv2d(C_in * 4, C_out, 1, 1, padding=0, dilation=dilation, groups=groups, bias=False, width_mult_list=width_mult_list)
        self.bn3 = USBatchNorm2d(C_out, width_mult_list)

        self.skip = USConv2d(C_in, C_out, 1, stride, padding=0, dilation=dilation, groups=1, bias=False,   width_mult_list=width_mult_list)
        self.bn4 = USBatchNorm2d(C_out, width_mult_list)

       

    def set_ratio(self, ratio):
        assert len(ratio) == 2
        self.ratio = ratio
        self.conv1.set_ratio((ratio[0], 1))
        self.bn1.set_ratio(1)
        self.conv2.set_ratio((1, 1))
        self.bn2.set_ratio(1)

        self.conv3.set_ratio((1, ratio[1]))
        self.bn3.set_ratio(ratio[1])

        if hasattr(self, 'skip'):
            self.skip.set_ratio(ratio)
            self.bn4.set_ratio(ratio[1])

    @staticmethod
    def _flops(h, w, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1):
        layer = MBCBlock(C_in, C_out, kernel_size, stride, dilation, groups=1, slimmable=False)
        flops, params = profile(layer, inputs=(torch.randn(1, C_in, h, w),))
        return flops

    @staticmethod
    def _latency(h, w, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1):
        layer = MBCBlock(C_in, C_out, kernel_size, stride, dilation, groups, slimmable=False)
        latency = compute_latency(layer, (1, C_in, h, w))
        return latency

    def forward_latency(self, size):
        c_in, h_in, w_in = size
        if self.slimmable:
            c_out = make_divisible(self.C_out * self.ratio[1])
        else:
            c_out = self.C_out
        if self.stride == 1:
            h_out = h_in;
            w_out = w_in
        else:
            h_out = h_in // 2;
            w_out = w_in // 2
        name = "FusedBlock_H%d_W%d_Cin%d_Cout%d_stride%d_dilation%d" % (
            h_in, w_in, c_in, c_out, self.stride, self.dilation)
        if name in latency_lookup_table:
            latency = latency_lookup_table[name]
        else:
            print("not found in latency_lookup_table:", name)
            latency = MBCBlock._latency(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.dilation,
                                        self.groups)
            latency_lookup_table[name] = latency
            np.save(table_file_name, latency_lookup_table)
        return latency, (c_out, h_out, w_out)

    def forward_flops(self, size, quantize=False):
        c_in, h_in, w_in = size
        if self.slimmable:
            c_out = make_divisible(self.C_out * self.ratio[1])
        else:
            c_out = self.C_out
        if self.stride == 1:
            h_out = h_in;
            w_out = w_in
        else:
            h_out = h_in // 2;
            w_out = w_in // 2
        name = "MBCBlock_H%d_W%d_Cin%d_Cout%d_kernel%d_stride%d" % (
            h_in, w_in, c_in, c_out, self.kernel_size, self.stride)
        if name in flops_lookup_table:
            flops = flops_lookup_table[name]
        else:
            print("not found in flops_lookup_table:", name)
            flops = MBCBlock._flops(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.dilation, self.groups)
            flops_lookup_table[name] = flops
            np.save(table_file_name, flops_lookup_table)
        # if quantize:
        #     ratio_dws = 3*3 / (3*3 + self.C_out)
        #     flops = ratio_dws * flops + (1-ratio_dws) * flops / 4

        return flops, (c_out, h_out, w_out)

    def forward(self, x):
        out = F.interpolate(x, size=(int(x.size(2))//2, int(x.size(3))//2), mode='bilinear', align_corners=True)
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        x_squeezed = self.squeeze(out)
        x_squeezed = self._se_reduce(x_squeezed)
        x_squeezed = self._swish(x_squeezed)
        x_squeezed = self._se_expand(x_squeezed)
        out = torch.sigmoid(x_squeezed) * out

        out = self.conv3(out)
        out = self.bn3(out)

        identity = self.bn4(self.skip(identity))

        out += identity
        if self.stride == 1:
            out = F.interpolate(out, size=(int(x.size(2)), int(x.size(3))), mode='bilinear', align_corners=True)
        out = self.relu(out)

        return out

class FusedBlock(nn.Module):
    def __init__(self, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1, slimmable=True,
                 width_mult_list=[1.]):
        super(FusedBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.slimmable = slimmable
        self.width_mult_list = width_mult_list
        assert stride in [1, 2]
        if self.stride == 2: self.dilation = 1
        self.ratio = (1., 1.)

        self.relu = nn.ReLU(inplace=True)




        self.conv1 = USConv2d(C_in, C_in * 4, 3, stride, padding=dilation, dilation=dilation, groups=groups,  bias=False, width_mult_list=width_mult_list)
        self.bn1 = USBatchNorm2d(C_in * 4, width_mult_list)

        num_squeezed_channels = max(1, int(C_in * 0.25))
        self.squeeze = nn.AdaptiveAvgPool2d((1,1))
        self._se_reduce = nn.Conv2d(C_in * 4, num_squeezed_channels, 1, 1,0)
        self._se_expand = nn.Conv2d(num_squeezed_channels, C_in * 4, 1, 1,0)
        self._swish = MemoryEfficientSwish()

        self.conv2 = USConv2d(C_in * 4, C_out, 1, 1, padding=0, dilation=dilation, groups=groups, bias=False, width_mult_list=width_mult_list)
        self.bn2 = USBatchNorm2d(C_out, width_mult_list)

        self.skip = USConv2d(C_in, C_out, 1, stride, padding=0, dilation=dilation, groups=1, bias=False,   width_mult_list=width_mult_list)
        self.bn3 = USBatchNorm2d(C_out, width_mult_list)

      

       
    def set_ratio(self, ratio):
        assert len(ratio) == 2
        self.ratio = ratio
        self.conv1.set_ratio((ratio[0], 1))
        self.bn1.set_ratio(1)
        self.conv2.set_ratio((1, ratio[1]))
        self.bn2.set_ratio(ratio[1])

        if hasattr(self, 'skip'):
            self.skip.set_ratio(ratio)
            self.bn3.set_ratio(ratio[1])

    @staticmethod
    def _flops(h, w, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1):
        layer = FusedBlock(C_in, C_out, kernel_size, stride, dilation, groups=1, slimmable=False)
        flops, params = profile(layer, inputs=(torch.randn(1, C_in, h, w),))
        return flops

    @staticmethod
    def _latency(h, w, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1):
        layer = FusedBlock(C_in, C_out, kernel_size, stride, dilation, groups, slimmable=False)
        latency = compute_latency(layer, (1, C_in, h, w))
        return latency

    def forward_latency(self, size):
        c_in, h_in, w_in = size
        if self.slimmable:
            c_out = make_divisible(self.C_out * self.ratio[1])
        else:
            c_out = self.C_out
        if self.stride == 1:
            h_out = h_in;
            w_out = w_in
        else:
            h_out = h_in // 2;
            w_out = w_in // 2
        name = "FusedBlock_H%d_W%d_Cin%d_Cout%d_stride%d_dilation%d" % (
            h_in, w_in, c_in, c_out, self.stride, self.dilation)
        if name in latency_lookup_table:
            latency = latency_lookup_table[name]
        else:
            print("not found in latency_lookup_table:", name)
            latency = FusedBlock._latency(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.dilation,
                                        self.groups)
            latency_lookup_table[name] = latency
            np.save(table_file_name, latency_lookup_table)
        return latency, (c_out, h_out, w_out)

    def forward_flops(self, size, quantize=False):
        c_in, h_in, w_in = size
        if self.slimmable:
            c_out = make_divisible(self.C_out * self.ratio[1])
        else:
            c_out = self.C_out
        if self.stride == 1:
            h_out = h_in;
            w_out = w_in
        else:
            h_out = h_in // 2;
            w_out = w_in // 2
        name = "FusedBlock_H%d_W%d_Cin%d_Cout%d_kernel%d_stride%d" % (
            h_in, w_in, c_in, c_out, self.kernel_size, self.stride)
        if name in flops_lookup_table:
            flops = flops_lookup_table[name]
        else:
            print("not found in flops_lookup_table:", name)
            flops = FusedBlock._flops(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.dilation, self.groups)
            flops_lookup_table[name] = flops
            np.save(table_file_name, flops_lookup_table)
        # if quantize:
        #     ratio_dws = 3*3 / (3*3 + self.C_out)
        #     flops = ratio_dws * flops + (1-ratio_dws) * flops / 4

        return flops, (c_out, h_out, w_out)

    def forward(self, x):
        out = F.interpolate(x, size=(int(x.size(2))//2, int(x.size(3))//2), mode='bilinear', align_corners=True)
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        x_squeezed = self.squeeze(out)
        x_squeezed = self._se_reduce(x_squeezed)
        x_squeezed = self._swish(x_squeezed)
        x_squeezed = self._se_expand(x_squeezed)
        out = torch.sigmoid(x_squeezed) * out

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.bn3(self.skip(identity))

        out += identity
        if self.stride == 1:
            out = F.interpolate(out, size=(int(x.size(2)), int(x.size(3))), mode='bilinear', align_corners=True)
        out = self.relu(out)

        return out

OPS = {
   # 'skip': lambda C_in, C_out, stride, slimmable, width_mult_list: FactorizedReduce(C_in, C_out, stride, slimmable, width_mult_list),
    'conv': lambda C_in, C_out, stride, slimmable, width_mult_list: ConvNorm(C_in, C_out, kernel_size=3, stride=stride, dilation=1, slimmable=slimmable, width_mult_list=width_mult_list),
    'conv_2x': lambda C_in, C_out, stride, slimmable, width_mult_list: BasicResidual2x(C_in, C_out, kernel_size=3,stride=stride, dilation=1,slimmable=slimmable, width_mult_list=width_mult_list),
    'dwconv': lambda C_in, C_out, stride, slimmable, width_mult_list: DWConv(C_in, C_out, kernel_size=3,  stride=stride, dilation=1,   slimmable=slimmable,  width_mult_list=width_mult_list),
    'dwsblock': lambda C_in, C_out, stride, slimmable, width_mult_list: DwsBlock(C_in, C_out, kernel_size=3, stride=stride, dilation=1,  slimmable=slimmable, width_mult_list=width_mult_list),
    'fusedblock': lambda C_in, C_out, stride, slimmable, width_mult_list: FusedBlock(C_in, C_out, kernel_size=3,  stride=stride, dilation=1,  slimmable=slimmable,  width_mult_list=width_mult_list),
     }
OPS_name = ["Conv3x3","BasicResidual2x","DwsBlock", "mbconv","FusedBlock"]
OPS_Class = OrderedDict()

#OPS_Class['skip'] = FactorizedReduce
OPS_Class['conv'] = ConvNorm
OPS_Class['fusedblock'] = FusedBlock
OPS_Class['conv_2x'] = BasicResidual2x
OPS_Class['dwsblock'] = DwsBlock
OPS_Class['dwconv'] = DWConv
