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

flags.DEFINE_integer('batch_size', 6, 'The number of images in each batch during training.')
flags.DEFINE_integer('classes', 20, 'The classes number.')
flags.DEFINE_integer('max_boxes_num', 80, 'The max number of boxes in one image.')
flags.DEFINE_string('saved_ckpt_path', './checkpoints/', 'Path to save training checkpoint.')

flags.DEFINE_float('initial_lr', 1e-2, 'The initial learning rate.')
flags.DEFINE_float('end_lr', 1e-6, 'The end learning rate.')
flags.DEFINE_integer('decay_steps', 40000, 'Used for poly learning rate.')
flags.DEFINE_float('weight_decay', 1e-4, 'The weight decay value for l2 regularization.')
flags.DEFINE_float('power', 0.9, 'Used for poly learning rate.')

with tf.name_scope('input'):
    x = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size, None, None, 3], name='x_input')
    y_true_1 = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size, 13, 13, 3, 5 + FLAGS.classes], name='y_true_13')
    y_true_2 = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size, 26, 26, 3, 5 + FLAGS.classes], name='y_true_26')
    y_true_3 = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size, 52, 52, 3, 5 + FLAGS.classes], name='y_true_52')


with tf.name_scope('yolov3'):
    yolo_outputs = YOLOV3(x, is_training=True, classes=FLAGS.classes)

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
        name=None
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

    saver = tf.train.Saver()

    # if os.path.exists(saved_ckpt_path):
    #ckpt = tf.train.get_checkpoint_state(FLAGS.saved_ckpt_path)
    #if ckpt and ckpt.model_checkpoint_path:
    #    saver.restore(sess, ckpt.model_checkpoint_path)
    #    print("Model restored...")

    for i in range(50000 + 1):
        batch_image, batch_y_true_1, batch_y_true_2, batch_y_true_3 = voc.__next__(batch_size=FLAGS.batch_size, max_boxes_num=FLAGS.max_boxes_num)
        feed_dict={
            x: batch_image,
            y_true_1: batch_y_true_1,
            y_true_2: batch_y_true_2,
            y_true_3: batch_y_true_3,
        }

        if i % 100 == 0:

            print("Step: %d, Loss: %f "%(i, sess.run(loss, feed_dict=feed_dict)))

        sess.run(train_op, feed_dict=feed_dict)

        if not os.path.exists('checkpoints'):
            os.mkdir('checkpoints')
        if i % 100 == 0:
            saver.save(sess, os.path.join('checkpoints', 'yolov3.model'), global_step=i)


