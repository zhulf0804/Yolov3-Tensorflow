#coding=utf-8

from __future__ import print_function
from __future__ import division

import tensorflow as tf
import os
import cv2
import numpy as np
from yolov3 import yolov3_body
from yolov3 import yolov3_loss as Loss
from datasets import Dataset

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 6, 'The number of images in each batch during training.')
flags.DEFINE_integer('classes', 20, 'The classes number.')
flags.DEFINE_integer('max_boxes_num', 80, 'The max number of boxes in one image.')
flags.DEFINE_string('restore_ckpt_path', './data/checkpoints', 'Path to save training checkpoint.')
flags.DEFINE_integer('max_steps', 50000, 'The max training steps.')
flags.DEFINE_integer('print_steps', 200, 'Used for print training information.')
flags.DEFINE_integer('saved_steps', 1000, 'Used for saving model.')

flags.DEFINE_string('saved_ckpt_path', './checkpoints', 'Path to save training checkpoint.')

flags.DEFINE_float('initial_lr', 1e-4, 'The initial learning rate.')
flags.DEFINE_float('end_lr', 1e-6, 'The end learning rate.')
flags.DEFINE_integer('decay_steps', 40000, 'Used for poly learning rate.')
flags.DEFINE_float('weight_decay', 1e-4, 'The weight decay value for l2 regularization.')
flags.DEFINE_float('power', 0.9, 'Used for poly learning rate.')

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

with tf.name_scope('input'):
    x = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size, None, None, 3], name='x_input')
    y_true_1 = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size, 13, 13, 3, 5 + FLAGS.classes], name='y_true_13')
    y_true_2 = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size, 26, 26, 3, 5 + FLAGS.classes], name='y_true_26')
    y_true_3 = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size, 52, 52, 3, 5 + FLAGS.classes], name='y_true_52')


with tf.variable_scope('yolov3'):
    yolo_outputs = yolov3_body(x, num_classes=FLAGS.classes, is_training=True)

variables = tf.contrib.framework.get_variables_to_restore(include=['yolov3'])

with tf.name_scope('loss'):
    loss = Loss(yolo_outputs=yolo_outputs, y_true=[y_true_1, y_true_2, y_true_3], num_classes=FLAGS.classes)

with tf.name_scope('learning_rate'):
    global_step = tf.Variable(0, trainable=False)
    lr = tf.train.polynomial_decay(
        learning_rate=FLAGS.initial_lr,
        global_step=global_step,
        decay_steps=FLAGS.decay_steps,
        end_learning_rate=FLAGS.end_lr,
        power=FLAGS.power,
        cycle=False,
        name='lr',
    )
    tf.summary.scalar('learning_rate', lr)



with tf.name_scope('opt'):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9).minimize(loss, global_step=global_step)

        #train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)

voc = Dataset('train')


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    exclude_vars = ['conv_59', 'conv_67', 'conv_75']


    variables_to_resotre = [v for v in variables if v.name.split('/')[1] not in exclude_vars]

    saver_to_restore = tf.train.Saver(variables_to_resotre)
    saver_to_save = tf.train.Saver()

    # if os.path.exists(saved_ckpt_path):
    ckpt = tf.train.get_checkpoint_state(FLAGS.restore_ckpt_path)
    if ckpt and ckpt.model_checkpoint_path:
        saver_to_restore.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")

    for i in range(FLAGS.max_steps + 1):
        batch_image, batch_y_true_1, batch_y_true_2, batch_y_true_3 = voc.__next__(batch_size=FLAGS.batch_size, max_boxes_num=FLAGS.max_boxes_num)

        feed_dict={
            x: batch_image,
            y_true_1: batch_y_true_1,
            y_true_2: batch_y_true_2,
            y_true_3: batch_y_true_3,
        }


        if i % FLAGS.print_steps == 0:

            print("Step: %d, Loss: %f "%(i, sess.run(loss, feed_dict=feed_dict)))

        sess.run(train_op, feed_dict=feed_dict)

        if not os.path.exists('checkpoints'):
            os.mkdir('checkpoints')
        if i % FLAGS.saved_steps == 0:
            saver_to_save.save(sess, os.path.join(FLAGS.saved_ckpt_path, 'yolov3.model'), global_step=i)


