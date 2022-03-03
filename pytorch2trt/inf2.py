import os
import torch
import tensorrt as trt
from PIL import Image
import numpy as np
import common
from test_segtrt2 import conver_engine
import time
import cv2
import glob

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
if __name__ == "__main__":
    onnx_file_path = './seg.onnx'
    engine_file_path = "./model_fp16_False_int8_False.trt"
    threshold = 0.5
    image_name = "./test.jpg"
    if not os.path.exists(engine_file_path):
        print("no engine file")
        # conver_engine(onnx_file_path, engine_file_path)
    print(f"Reading engine from file {engine_file_path}")
    preprocess_time = 0
    process_time = 0
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        with runtime.deserialize_cuda_engine(f.read()) as engine, engine.create_execution_context() as context:
            inputs, outputs, bindings, stream = common.allocate_buffers(engine)
            image = cv2.imread(image_name)
            a = time.time()
            image_height, image_width = image.shape[:2]
            # image = cv2.resize(image, (768, 768)).transpose((2, 0, 1))
            image = np.array(cv2.resize(image, (768, 768)), dtype=np.float)
            image -= np.array([102.9801, 115.9465, 122.7717])
            image = np.transpose(image, (2, 0, 1)).ravel()
            # image_batch = np.stack([image], 0).ravel()
            np.copyto(inputs[0].host, image)
            preprocess_time += time.time() - a
            a = time.time()
            trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            process_time += time.time() - a
            x = trt_outputs[0].reshape((100, 5))
            # imshow
            image = cv2.imread(image_name)
            indices = x[:, -1] > threshold
            polygons = x[indices, :-1]
            scores = x[indices, -1]
            polygons[:, ::2] *= 1. * image.shape[1] / 768
            polygons[:, 1::2] *= 1. * image.shape[0] / 768

            for polygon, score in zip(polygons, scores):
                print(polygon, score)
                cv2.rectangle(image, (int(polygon[0]), int(polygon[1])), (int(polygon[2]), int(polygon[3])), color=(0, 255, 0), thickness=2)
                cv2.putText(image, str("%.3f" % score), (int(polygon[0]), int(polygon[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, False)
            cv2.imwrite("tensorrt_demo.jpg", image)

            print("preprocess time: ", preprocess_time, ", inference time: ", process_time)

