#coding=utf-8

from __future__ import print_function
from __future__ import division

import cv2
import tensorflow as tf
import numpy as np

from yolov3 import yolov3_body
from yolov3 import yolov3_head
from yolov3 import yolo_eval
from utils.get_anchors import get_anchors


def image_preprocess(image, target_size=[416, 416]):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    h, w, _ = image.shape
    t_h, t_w = target_size

    scale = min(t_h / h, t_w / w)

    n_w, n_h = int(scale * w), int(scale * h)

    image_resized = cv2.resize(image, (n_w, n_h))

    image_padded = np.full(shape=[t_h, t_w, 3], fill_value=128.0)

    dw, dh = (t_w - n_w) // 2, (t_h - n_h) // 2

    image_padded[dh:n_h + dh, dw:n_w + dw, :] = image_resized
    image_padded = image_padded / 255.
    image_padded = np.expand_dims(image_padded, 0)
    image_padded = tf.constant(image_padded)
    image_padded = tf.cast(image_padded, tf.float32)
    return image_padded

def detect_image(image_path):
    img = cv2.imread(image_path)
    height, width, _ = img.shape

    img = image_preprocess(img)
    yolo_outputs = yolov3_body(inputs=img, is_training=False, classes=20)
    #anchors = get_anchors()

    boxes_, scores_, classes_ = yolo_eval(yolo_outputs, num_classes=20, image_shape=(height, width), max_boxes=30, score_threshold=.6, iou_threshold=.5)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        boxes = sess.run(boxes_)

        for bbox in boxes:
            print(bbox)
            image = cv2.rectangle(img, (int(float(bbox[0])), int(float(bbox[1]))), (int(float(bbox[2])), int(float(bbox[3]))), (255, 0, 255), 2)


if __name__ == '__main__':
    detect_image(image_path='utils/2008_000289.jpg')