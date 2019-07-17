#coding=utf-8

from __future__ import print_function
from __future__ import division

import tensorflow as tf
import os
import numpy as np
from yolov3 import yolov3_body as YOLOV3
from yolov3 import yolov3_loss as Loss
from datasets import Dataset

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 4, 'The number of images in each batch during training.')
flags.DEFINE_integer('classes', 20, 'The classes number.')
flags.DEFINE_integer('max_boxes_num', 20, 'The max number of boxes in one image.')


with tf.name_scope('input'):
    x = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size, None, None, 3], name='x_input')
    y_true_1 = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size, 13, 13, 3, 5 + FLAGS.classes], name='y_true_13')
    y_true_2 = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size, 26, 26, 3, 5 + FLAGS.classes], name='y_true_26')
    y_true_3 = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size, 52, 52, 3, 5 + FLAGS.classes], name='y_true_52')


with tf.name_scope('yolov3'):
    yolo_outputs = YOLOV3(x, is_training=True, classes=FLAGS.classes)

with tf.name_scope('loss'):
    loss = Loss(yolo_outputs=yolo_outputs, y_true=[y_true_1, y_true_2, y_true_3], num_classes=FLAGS.classes)

with tf.name_scope('opt'):
    train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)

voc = Dataset('train')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    for i in range(20000):
        batch_image, batch_y_true_1, batch_y_true_2, batch_y_true_3 = voc.__next__(batch_size=FLAGS.batch_size, max_boxes_num=FLAGS.max_boxes_num)
        feed_dict={
            x: batch_image,
            y_true_1: batch_y_true_1,
            y_true_2: batch_y_true_2,
            y_true_3: batch_y_true_3,
        }

        print("Step: %d, Loss: %f "%(i, sess.run(loss, feed_dict=feed_dict)))

        sess.run(train_op, feed_dict=feed_dict)

        if not os.path.exists('checkpoints'):
            os.mkdir('checkpoints')
        if i % 100 == 0:
            saver.save(sess, os.path.join('checkpoints', 'yolov3.model'), global_step=i)


