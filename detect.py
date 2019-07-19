#coding=utf-8

from __future__ import print_function
from __future__ import division

import cv2
import tensorflow as tf
import numpy as np
import os

from yolov3 import yolov3_body
from yolov3 import yolo_eval
from utils.get_anchors import get_anchors

from PIL import Image


def letterbox_resize(img, new_width, new_height, interp=0):
    '''
    Letterbox resize. keep the original aspect ratio in the resized image.
    '''
    ori_height, ori_width = img.shape[:2]

    resize_ratio = min(new_width / ori_width, new_height / ori_height)

    resize_w = int(resize_ratio * ori_width)
    resize_h = int(resize_ratio * ori_height)

    img = cv2.resize(img, (resize_w, resize_h), interpolation=interp)
    image_padded = np.full((new_height, new_width, 3), 128, np.uint8)

    dw = int((new_width - resize_w) / 2)
    dh = int((new_height - resize_h) / 2)

    image_padded[dh: resize_h + dh, dw: resize_w + dw, :] = img

    return image_padded

def detect_image(image_path, i=1):
    img_ = cv2.imread(image_path)
    height, width, _ = img_.shape
    img = letterbox_resize(img_, 416, 416)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.asarray(img, np.float32)
    img = img[np.newaxis, :] / 255.

    input_data = tf.placeholder(tf.float32, [1, 416, 416, 3], name='input_data')


    with tf.variable_scope('yolov3'):

        yolo_outputs = yolov3_body(inputs=input_data, num_classes=80, is_training=False)
    #anchors = get_anchors()

    #with tf.variable_scope('yolov3'):

        #feats = Yolov3(inputs, True, num_classes)
        #yolo_outputs = yolo_inference(input_data, 3, 80)



    boxes_, scores_, classes_ = yolo_eval(yolo_outputs, num_classes=80, image_shape=(height, width), max_boxes=80, score_threshold=.6, iou_threshold=.5)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()

        #saver.restore(sess, './data/checkpoints/yolov3.ckpt')

        ckpt = tf.train.get_checkpoint_state('./checkpoints')
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        boxes = sess.run(boxes_, feed_dict={input_data: img})

        print(boxes)

        print(len(boxes))

        for bbox in boxes:
            print(bbox)
            img_ = cv2.rectangle(img_, (int(float(bbox[1])), int(float(bbox[0]))), (int(float(bbox[3])), int(float(bbox[2]))), (255, 0, 255), 2)

        if not os.path.exists('output'):
            os.mkdir('output')



        cv2.imwrite('output/%d.png'%i, img_)

if __name__ == '__main__':
    #detect_image(image_path='utils/COCO_test2014_000000000069.jpg')
    detect_image(image_path='utils/2007_001311.jpg')