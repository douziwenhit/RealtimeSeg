import pycuda.autoinit
import numpy as np
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
import onnx
from model_seg import Network_Multi_Path_Infer as Network
import os.path as osp
from matplotlib import pyplot as plt
import matplotlib
from test import SegTester
import torchvision.transforms as T
from tools.datasets import Cityscapes
from tools.datasets import Cityscapes
matplotlib.use( 'tkagg' )
import shutil
import collections
filename = '/home/dou/Documents/code/mycode/pytorch2trt/bremen_000001_000019_leftImg8bit.png'
max_batch_size = 1
onnx_model_path = '/home/dou/Documents/code/mycode/pytorch2trt/seg.onnx'

TRT_LOGGER = trt.Logger()  # This logger is required to build an engine
#//dataset//////////////////////
"""Data Dir"""
dataset_path = "/home/dou/Documents/pytorch-multigpu/cityscapes"
img_root_folder = dataset_path
gt_root_folder = dataset_path
train_source = osp.join(dataset_path, "cityscapes_train_fine.txt")
train_eval_source = osp.join(dataset_path, "cityscapes_train_val_fine.txt")
eval_source = osp.join(dataset_path, "cityscapes_val_fine.txt")
test_source = osp.join(dataset_path, "cityscapes_test.txt")

data_setting = {'img_root': config.img_root_folder,
                        'gt_root': config.gt_root_folder,
                        'train_source': config.train_eval_source,
                        'eval_source': config.eval_source,
                        'test_source': config.test_source,
                        'down_sampling': config.down_sampling}



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
    image_cv = cv2.resize(image_cv, (1024, 2048))
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


TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
def GiB(val):
    return val * 1 << 30

def get_engine(max_batch_size=1, onnx_file_path="", engine_file_path="", \
               fp16_mode=False, int8_mode=False, save_engine=False,
               ):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

    def build_engine(max_batch_size, save_engine):
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        with trt.Builder(TRT_LOGGER) as builder, \
                builder.create_network(explicit_batch) as network, \
                trt.OnnxParser(network, TRT_LOGGER) as parser:

            builder.max_workspace_size = 1 << 30  # Your workspace size
            builder.max_batch_size = max_batch_size
            # pdb.set_trace()
            builder.fp16_mode = fp16_mode  # Default: False
            builder.int8_mode = int8_mode  # Default: False
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
        T.Resize((1024,2048)), 
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        
    ])
    
    #img = Image.open(path)
    img_tensor = trans(path)
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

dataset = Cityscapes(data_setting, 'test', None)
save_path = "%dx%d_trt"%(1024, 2048)
create_exp_dir(save_path)
# These two modes are dependent on hardwares
fp16_mode = False
int8_mode = False
trt_engine_path = './model_fp16_{}_int8_{}.trt'.format(fp16_mode, int8_mode)
# Build an engine
engine = get_engine(1, onnx_model_path, trt_engine_path, fp16_mode, int8_mode)
# Create the context for this engine
context = engine.create_execution_context()
# Allocate buffers for input and output
inputs, outputs, bindings, stream = allocate_buffers(engine) # input, output: host # bindings

# Do inference
shape_of_output = (1,19,1024,2048)
# Load data to the buffer
cap = cv2.VideoCapture(0)


while(1):
    # get a frame
    ret, frame = cap.read()
    # show a frame
    
    cv2.imshow("capture", frame)
    image = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))  
    img_scale = load_data(image)
    inputs[0].host = img_scale

     # inputs[1].host = ... for multiple input
    t1 = time.time()
    trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream) # numpy data
    t2 = time.time()
    feat = postprocess_the_outputs(trt_outputs[0], shape_of_output)
    feat = feat[0].transpose(1, 2, 0)
    pred = feat.argmax(2).astype(np.uint64)
    #####显示分割后的图片###########################################
    colors = dataset.get_class_colors()
    image = image.resize((2048,1024),Image.ANTIALIAS)
    comp_img = show_prediction(colors, -1, image, pred)
    cv2.imshow("test.png", comp_img[:,:,::-1])
    for x in range(pred.shape[0]):
        for y in range(pred.shape[1]):
            pred[x, y] = cityscapes_trainID2id[pred[x, y]]

    print("Inference time with the TensorRT engine: {}".format(t2-t1))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows() 

img_scale = load_data('/home/dou/Documents/code/mycode/pytorch2trt/bremen_000001_000019_leftImg8bit.png')

inputs[0].host = img_scale

# inputs[1].host = ... for multiple input
t1 = time.time()
trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream) # numpy data
t2 = time.time()
feat = postprocess_the_outputs(trt_outputs[0], shape_of_output)
feat = feat[0].transpose(1, 2, 0)
pred = feat.argmax(2).astype(np.uint64)
#####显示分割后的图片###########################################
colors = dataset.get_class_colors()
image = cv2.imread('/home/dou/Documents/code/mycode/pytorch2trt/bremen_000001_000019_leftImg8bit.png')
comp_img = show_prediction(colors, -1, image, pred)
cv2.imwrite("test.png", comp_img[:,:,::-1])

for x in range(pred.shape[0]):
    for y in range(pred.shape[1]):
        pred[x, y] = cityscapes_trainID2id[pred[x, y]]
cv2.imwrite("test1.png", pred)
print('TensorRT ok')
print("Inference time with the TensorRT engine: {}".format(t2-t1))
print('All completed!')
