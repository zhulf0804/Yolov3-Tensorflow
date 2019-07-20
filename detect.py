#coding=utf-8

from __future__ import print_function
from __future__ import division

import cv2
import tensorflow as tf
import numpy as np
import os
import colorsys

import config

from yolov3 import yolov3_body
from yolov3 import yolo_eval
from utils.get_anchors import get_anchors

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('image_path', 'utils/2007_001311.jpg', 'Path to save training checkpoint.')
flags.DEFINE_string('ckpt_path', config._coco_tf_weights, 'Path to save training summary.')

flags.DEFINE_string('names_path', config._coco_names, 'Path to save training summary.')
flags.DEFINE_integer('num_classes', config._coco_classes, 'The classes number.')



with open(FLAGS.names_path, 'r') as f:
    classmap = f.readlines()

#colormap = config._coco_colormap

# Generate colors for drawing bounding boxes.
hsv_tuples = [(x / FLAGS.num_classes, 1., 1.)
              for x in range(FLAGS.num_classes)]
colormap = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
colormap = list(
    map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colormap))
np.random.seed(10101)  # Fixed seed for consistent colors across runs.
np.random.shuffle(colormap)  # Shuffle colors to decorrelate adjacent classes.
np.random.seed(None)  # Reset seed to default.

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
    img = letterbox_resize(img_, config._input_size, config._input_size)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.asarray(img, np.float32)
    img = img[np.newaxis, :] / 255.

    input_data = tf.placeholder(tf.float32, [1, config._input_size, config._input_size, 3], name='input_data')


    with tf.variable_scope('yolov3'):

        yolo_outputs = yolov3_body(inputs=input_data, num_classes=FLAGS.num_classes, is_training=False)


    boxes_, scores_, classes_ = yolo_eval(yolo_outputs, num_classes=FLAGS.num_classes, image_shape=(height, width), max_boxes=config._max_boxes, score_threshold=.6, iou_threshold=.5)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()

        #saver.restore(sess, './data/checkpoints/yolov3.ckpt')

        ckpt = tf.train.get_checkpoint_state(FLAGS.ckpt_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Model restored')

        boxes, scores, ids = sess.run([boxes_, scores_, classes_], feed_dict={input_data: img})

        #print(boxes)

        for i, bbox in enumerate(boxes):
            img_ = cv2.rectangle(img_, (int(float(bbox[1])), int(float(bbox[0]))), (int(float(bbox[3])), int(float(bbox[2]))), colormap[int(ids[i])], 2)
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(img_, classmap[int(ids[i])].strip(), (int(float(bbox[1])), int(float(bbox[0])) - 7), font, 0.5, (255, 255, 0), 1)
        if not os.path.exists('output'):
            os.mkdir('output')

        basename = os.path.basename(image_path).split('.')[0]

        cv2.imwrite('output/%s_pred.png'%basename, img_)

        if len(boxes) == 0:
            print("no object is detected.")
        elif len(boxes) == 1:
            print("one object is detected.")
            print(classmap[int(ids[0])].strip(), ": ",  scores[0])
        else:
            print("%d objects are detected." %len(boxes))
            for i in range(len(boxes)):
                print(classmap[ids[i]].strip(), ": ", scores[i])



        print("prediction file saved in output/%s_pred.png" %basename)

if __name__ == '__main__':
    #detect_image(image_path='utils/COCO_test2014_000000000069.jpg')
    detect_image(FLAGS.image_path)