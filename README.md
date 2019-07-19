## Introduction

A Tensorflow Implementation of Yolov3.

## Pretrained weights Prepare

The pretrained darknet weights file can be downloaded [here](https://pjreddie.com/media/files/yolov3.weights). Place this weights file under directory **./data** and then run:

> python convert_weights.py

## Detect

> python detect.py --image_path utils/2008_000289.jpg

## Train

> python train.py



## Reference
+ [https://github.com/qqwweee/keras-yolo3](https://github.com/qqwweee/keras-yolo3)
+ [https://github.com/YunYang1994/tensorflow-yolov3](https://github.com/YunYang1994/tensorflow-yolov3)
+ [https://github.com/wizyoung/YOLOv3_TensorFlow](https://github.com/wizyoung/YOLOv3_TensorFlow)
+ [https://github.com/aloyschen/tensorflow-yolo3](https://github.com/aloyschen/tensorflow-yolo3)