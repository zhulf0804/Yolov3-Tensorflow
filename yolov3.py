#coding=utf-8

from __future__ import print_function
from __future__ import division


import tensorflow as tf
import math
import os
import numpy as np
from utils.get_anchors import get_anchors

import tensorflow as tf
import math
import os
import numpy as np

#Used for BN
_BATCH_NORM_DECAY = 0.99
_BATCH_NORM_EPSILON = 1e-3

# number_params in yolov3.weights:     62001757
# number_params in our darknet53:      30122592(voc: 20), 62001757(coco: 80)
# number_params in our yolo model:     51180609
# number_variables in our yolo model:  366


def batch_norm(inputs, is_training):

  bn_layer = tf.layers.batch_normalization(inputs=inputs,
                                           momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
                                           scale=True, training=is_training)
  return tf.nn.leaky_relu(bn_layer, alpha=0.1)

def convolutional_(inputs, filters_num, kernel_size, stride, is_training, name, use_bias=False, bn=True):
    net = tf.layers.conv2d(
        inputs=inputs, filters=filters_num,
        kernel_size=kernel_size, strides=[stride, stride], kernel_initializer=tf.glorot_uniform_initializer(),
        padding=('SAME' if stride == 1 else 'VALID'),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=5e-4), use_bias=use_bias, name=name)


    if bn:
        net = batch_norm(net, is_training=is_training)

    return net



def residual(inputs, num_featues, is_training, conv_index):

    shortcut = inputs
    net = convolutional_(inputs, num_featues // 2, 1, 1, is_training, name='conv_%d'%conv_index, use_bias=False, bn=True)
    conv_index += 1
    net = convolutional_(net, num_featues, 3, 1, is_training, name='conv_%d'%conv_index, use_bias=False, bn=True)
    conv_index += 1

    return shortcut + net, conv_index

def _darknet53(inputs, conv_index, is_training=True, norm_decay=0.99, norm_epsilon=1e-3):

    with tf.name_scope('init_conv'):
        net = convolutional_(inputs, 32, 3, 1, is_training, name='conv_%d'%conv_index)
        conv_index += 1

    with tf.name_scope('convs_1'):
        net = tf.pad(net, paddings=[[0, 0], [1, 0], [1, 0], [0, 0]], mode='CONSTANT')
        net = convolutional_(net, 64, 3, 2, is_training, name='conv_%d'%conv_index)

        conv_index += 1

        for i in range(1):
            with tf.name_scope("conv_%d"%i):
                net, conv_index = residual(net, 64, is_training, conv_index)

    with tf.name_scope('convs_2'):
        net = tf.pad(net, paddings=[[0, 0], [1, 0], [1, 0], [0, 0]], mode='CONSTANT')
        net = convolutional_(net, 128, 3, 2, is_training, name='conv_%d'%conv_index)
        conv_index += 1
        for i in range(2):
            with tf.name_scope('conv_%d'%i):
                net, conv_index = residual(net, 128, is_training, conv_index)


    with tf.name_scope('convs_3'):
        net = tf.pad(net, paddings=[[0, 0], [1, 0], [1, 0], [0, 0]], mode='CONSTANT')
        net = convolutional_(net, 256, 3, 2, is_training, name='conv_%d'%conv_index)
        conv_index += 1
        for i in range(8):
            with tf.name_scope('conv_%d'%i):
                net, conv_index = residual(net, 256, is_training, conv_index)

    route_1 = net



    with tf.name_scope('convs_4'):
        net = tf.pad(net, paddings=[[0, 0], [1, 0], [1, 0], [0, 0]], mode='CONSTANT')
        net = convolutional_(net, 512, 3, 2, is_training, name='conv_%d'%conv_index)
        conv_index += 1
        for i in range(8):
            with tf.name_scope('conv_%d'%i):
                net, conv_index = residual(net, 512, is_training, conv_index)

    route_2 = net

    with tf.name_scope('convs_5'):

        net = tf.pad(net, paddings=[[0, 0], [1, 0], [1, 0], [0, 0]], mode='CONSTANT')
        net = convolutional_(net, 1024, 3, 2, is_training, name='conv_%d'%conv_index)
        conv_index += 1

        for i in range(4):
            with tf.name_scope('conv_%d'%i):
                net, conv_index = residual(net, 1024, is_training, conv_index)

    return route_1, route_2, net, conv_index



def yolov3_body(inputs, num_classes, is_training=True):

    conv_index = 1
    route_1, route_2, net, conv_index = _darknet53(inputs, conv_index, is_training=is_training, norm_decay=_BATCH_NORM_DECAY, norm_epsilon=_BATCH_NORM_EPSILON)

    with tf.name_scope("large_obj_conv"):

        net = convolutional_(net, 512,1, 1, is_training, name='conv_%d' % conv_index)
        conv_index += 1

        net = convolutional_(net, 1024, 3, 1, is_training, name='conv_%d' % conv_index)
        conv_index += 1

        net = convolutional_(net, 512, 1, 1, is_training, name='conv_%d' % conv_index)
        conv_index += 1

        net = convolutional_(net, 1024, 3, 1, is_training, name='conv_%d' % conv_index)
        conv_index += 1

        net = convolutional_(net, 512, 1, 1, is_training, name='conv_%d' % conv_index)
        conv_index += 1

        conv_lobj_branch = convolutional_(net, 1024, 3, 1, is_training, name='conv_%d' % conv_index)
        conv_index += 1

        y1 = convolutional_(conv_lobj_branch, 3*(num_classes + 5), 1, 1, is_training, name='conv_%d' % conv_index, use_bias=True, bn=False)
        conv_index += 1

    with tf.name_scope('middle_obj_conv'):
        with tf.name_scope('up_sample_conv'):

            net = convolutional_(net, 256, 1, 1, is_training, name='conv_%d' % conv_index)
            conv_index += 1
            net = tf.image.resize_nearest_neighbor(net, tf.shape(net)[1:3]*2)

        net = tf.concat([net, route_2], axis=-1)

        net = convolutional_(net, 256, 1, 1, is_training, name='conv_%d' % conv_index)
        conv_index += 1
        net = convolutional_(net, 512, 3, 1, is_training, name='conv_%d' % conv_index)
        conv_index += 1
        net = convolutional_(net, 256, 1, 1, is_training, name='conv_%d' % conv_index)
        conv_index += 1
        net = convolutional_(net, 512, 3, 1, is_training, name='conv_%d' % conv_index)
        conv_index += 1
        net = convolutional_(net, 256, 1, 1, is_training, name='conv_%d' % conv_index)
        conv_index += 1
        conv_mobj_branch = convolutional_(net, 512, 3, 1, is_training, name='conv_%d' % conv_index)
        conv_index += 1

        y2 = convolutional_(conv_mobj_branch, 3 * (num_classes + 5), 1, 1, is_training, name='conv_%d' % conv_index,
                            use_bias=True, bn=False)
        conv_index += 1


    with tf.name_scope('small_obj_conv'):
        with tf.name_scope('up_sample_conv'):

            net = convolutional_(net, 128, 1, 1, is_training, name='conv_%d' % conv_index)
            conv_index += 1
            net = tf.image.resize_nearest_neighbor(net, tf.shape(net)[1:3]*2)

        net = tf.concat([net, route_1], axis=-1)

        net = convolutional_(net, 128, 1, 1, is_training, name='conv_%d' % conv_index)
        conv_index += 1
        net = convolutional_(net, 256, 3, 1, is_training, name='conv_%d' % conv_index)
        conv_index += 1

        net = convolutional_(net, 128, 1, 1, is_training, name='conv_%d' % conv_index)
        conv_index += 1
        net = convolutional_(net, 256, 3, 1, is_training, name='conv_%d' % conv_index)
        conv_index += 1

        net = convolutional_(net, 128, 1, 1, is_training, name='conv_%d' % conv_index)
        conv_index += 1
        conv_sobj_branch = convolutional_(net, 256, 3, 1, is_training, name='conv_%d' % conv_index)
        conv_index += 1

        y3 = convolutional_(conv_sobj_branch, 3 * (num_classes + 5), 1, 1, is_training, name='conv_%d' % conv_index,
                            use_bias=True, bn=False)
        conv_index += 1

    return y1, y2, y3


def yolov3_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    # Convert final layer features to bounding box parameters
    num_anchors = len(anchors)

    #Reshape to batch, height, width, num_anchors, box_params
    anchors_tensor = tf.reshape(tf.constant(anchors), (1, 1, 1, num_anchors, 2))

    grid_shape = tf.shape(feats)[1:3] # height, width
    grid_y = tf.tile(tf.reshape(tf.range(grid_shape[0]), [-1, 1, 1, 1]), [1, grid_shape[1], 1, 1])
    grid_x = tf.tile(tf.reshape(tf.range(grid_shape[1]), [1, -1, 1, 1]), [grid_shape[0], 1, 1, 1])

    grid = tf.concat([grid_x, grid_y], axis=-1)

    grid = tf.cast(grid, tf.float32)

    feats = tf.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # Adjust predictions to each spatial grid point and anchor size.
    box_xy = (tf.nn.sigmoid(feats[..., :2]) + grid) / tf.cast(grid_shape[::-1], tf.float32)
    box_wh = tf.exp(feats[..., 2:4]) * tf.cast(anchors_tensor, tf.float32) / tf.cast(input_shape[::-1], tf.float32)
    box_cofidence = tf.nn.sigmoid(feats[..., 4:5])
    box_class_probs = tf.nn.sigmoid(feats[..., 5:])

    if calc_loss:
        return grid, feats, box_xy, box_wh

    return box_xy, box_wh, box_cofidence, box_class_probs

def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    '''
    Get corrected boxes
    :param box_xy:
    :param box_wh:
    :param input_shape:
    :param image_shape:
    :return:
    '''

    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = tf.cast(input_shape, tf.float32)
    image_shape = tf.cast(image_shape, tf.float32)

    new_shape = tf.round(image_shape * tf.reduce_min(input_shape / image_shape))

    offset = (input_shape - new_shape)  / 2. / input_shape

    scale = input_shape / new_shape
    box_yx = (box_yx - offset) * scale

    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)

    boxes = tf.concat([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)

    boxes *= tf.concat([image_shape, image_shape], axis=-1)


    return boxes


def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    box_xy, box_wh, box_confidence, box_class_probs = yolov3_head(feats, anchors, num_classes, input_shape)
    #print(box_xy)
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)

    boxes = tf.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = tf.reshape(box_scores, [-1, num_classes])

    return boxes, box_scores

def yolo_eval(yolo_outputs, num_classes, image_shape, max_boxes=30, score_threshold=.6, iou_threshold=.5):
    '''
    Evaluate YOLO model on given input and returen filtered boxes.
    :param yolo_outputs:
    :param anchors:
    :param num_classes:
    :param image_shape:
    :param max_boxes:
    :param score_threshold:
    :param iou_threshold:
    :return:
    '''

    #anchors = get_anchors()
    anchors = np.array([[10,13], [16,30], [33,23], [30,61], [62,45], [59,119], [116,90], [156,198], [373,326]])

    num_layers = len(yolo_outputs)
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    input_shape = tf.shape(yolo_outputs[0])[1:3] * 32

    boxes = []
    box_scores = []

    for l in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l], anchors[anchor_mask[l]], num_classes, input_shape, image_shape)

        boxes.append(_boxes)
        box_scores.append(_box_scores)

    boxes = tf.concat(boxes, axis=0)
    box_scores = tf.concat(box_scores, axis=0)

    mask = box_scores >= score_threshold

    max_boxes_tensor = tf.constant(max_boxes, dtype=tf.int32)

    boxes_ = []
    scores_ = []
    classes_ = []

    for c in range(num_classes):
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)

        class_boxes = tf.gather(class_boxes, nms_index)
        class_box_scores = tf.gather(class_box_scores, nms_index)

        classes = tf.ones_like(class_box_scores, tf.int32) * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)

    boxes_ = tf.concat(boxes_, axis=0)
    scores_ = tf.concat(scores_, axis=0)
    classes_ = tf.concat(classes_, axis=0)

    return boxes_, scores_, classes_


def box_iou(b1, b2):
    '''

    :param b1: Shape=(13, 13, 3, 4)
    :param b2: Shape=(n, 4)
    :return:
    '''
    # Expand dim to apply broadcasting
    b1 = tf.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting
    b2 = tf.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = tf.maximum(b1_mins, b2_mins)
    intersect_maxes = tf.minimum(b1_maxes, b2_maxes)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)

    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]

    iou = intersect_area / (b1_area + b2_area - intersect_area)

    #print(iou.shape)

    return iou

def yolov3_loss(yolo_outputs, y_true, num_classes, ignore_thresh=.5):
    '''

    :param yolo_outputs: [y1, y2, y3], yi_Shape=(batch_size, grid_shape, grid_shape, 3 * (5 + num_classes))
    :param y_true: [g1, g2, g2], gi_Shape=(batch_size, grid_shape, grid_shape, 3, 5 + num_classes)
    :param anchors: Shape=(9, 2)
    :param num_classes:
    :param ignore_thresh:
    :return:
    '''
    anchors = get_anchors()
    num_layers = len(anchors) // 3
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    input_shape = tf.cast(tf.shape(yolo_outputs[0])[1:3] * 32, tf.float32)

    grid_shapes = [tf.cast(tf.shape(yolo_outputs[l])[1:3], tf.float32) for l in range(num_layers)]


    loss = 0
    m = tf.shape(yolo_outputs[0])[0] # batch_size
    mf = tf.cast(m, tf.float32)

    for l in range(num_layers):
        object_mask = y_true[l][..., 4:5]  # Shape = (batch_size, grid_shape, grid_shape, 3, 1)
        true_class_probs = y_true[l][..., 5:] # Shape = (batch_size, grid_shape, grid_shape, 3, 20)

        grid, raw_pred, pred_xy, pred_wh = yolov3_head(yolo_outputs[l], anchors[anchor_mask[l]], num_classes, input_shape, calc_loss=True)

        # grid: Shape=(13, 13, 1, 2)
        # raw_pred: Shape=(4, 13, 13, 3, 25)
        # pred_xy: Shape=(4, 13, 13, 3, 2)  [0, 1]
        # pred_wh: Shape=(4, 13, 13, 3, 2)  [0, 1]


        pred_box = tf.concat([pred_xy, pred_wh], axis=-1) # Shape=(4, 13, 13, 3, 4)


        # Darknet raw box to calculate loss
        raw_true_xy = y_true[l][..., :2] * grid_shapes[l][::-1] - grid  # (batch_size, 13, 13, 3, 2) * (2, ) - (13, 13, 1, 2) => (batch_size, 13, 13, 3, 2)

        raw_true_wh = tf.math.log(tf.clip_by_value(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1], 1e-9, 1e9)) # (batch_size, 13, 13, 3, 2) / (3, 2) * (2, 1) => (batch_size, 13, 13, 3, 2)

        object_mask_bool = tf.cast(object_mask, tf.bool) # Shape = (batch_size, grid_shape, grid_shape, 3, 1)

        raw_true_wh = tf.where(tf.concat([object_mask_bool, object_mask_bool], axis=-1), raw_true_wh, tf.zeros_like(raw_true_wh))


        box_loss_scale = 2 - y_true[l][..., 2:3] * y_true[l][..., 3:4]

        # Find ignore mask, iterate over each of batch

        ignore_mask = tf.TensorArray(tf.float32, size=1, dynamic_size=True)

        def loop_body(b, ignore_mask):
            true_box = tf.boolean_mask(y_true[l][b, ..., 0:4], object_mask_bool[b, ..., 0])  # tf.boolean_mask(Shape=(13, 13, 3, 4), Shape=(13, 13, 3))
            iou = box_iou(pred_box[b], true_box)  # pred_box[b]: Shape(13, 13, 3, 4), true_box: Shape(?, ?)
            best_iou = tf.reduce_max(iou, axis=-1)
            ignore_mask = ignore_mask.write(b, tf.cast(best_iou < ignore_thresh, tf.float32))
            return b + 1, ignore_mask
        b = tf.constant(0)
        _, ignore_mask = tf.while_loop(lambda b, ignore_mask: b < m, loop_body, [b, ignore_mask])    # Need

        ignore_mask = ignore_mask.stack()
        ignore_mask = tf.expand_dims(ignore_mask, -1) # Shape=(?, 13, 13, 3, 1)

        #print(box_loss_scale.shape)
        # object_mask: Shape=(4, 13, 13, 3, 1), box_loss_scale: Shape=(4, 13, 13, 3, 1), raw_true_xy: Shape=(4, 13, 13, 3, 2), raw_pred[..., 0:2]: Shape=(4, 13, 13, 3, 2)
        xy_loss = object_mask * box_loss_scale * tf.nn.sigmoid_cross_entropy_with_logits(labels=raw_true_xy, logits=raw_pred[..., 0:2])
        wh_loss = object_mask * box_loss_scale * 0.5 * tf.square(raw_true_wh - raw_pred[..., 2:4])

        confidence_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=raw_pred[..., 4:5]) + \
                          (1 - object_mask) * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=raw_pred[..., 4:5]) * ignore_mask

        class_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=true_class_probs, logits=raw_pred[..., 5:])

        xy_loss = tf.reduce_sum(xy_loss) / mf
        wh_loss = tf.reduce_sum(wh_loss) / mf
        confidence_loss = tf.reduce_sum(confidence_loss) / mf
        class_loss = tf.reduce_sum(class_loss) / mf

        loss += xy_loss + wh_loss + confidence_loss + class_loss

    return loss

def ignore_test():
    inputs = tf.constant(0.5, shape=[4, 416, 416, 3])

    with tf.variable_scope('yolov3'):
        a, b, c = yolov3_body(inputs, num_classes=80, is_training=True)

    variables = tf.trainable_variables()

    exclude_vars = ['conv_59', 'conv_67', 'conv_75']

    variables_to_resotre = [v for v in variables if v.name.split('/')[1] not in exclude_vars]

    print(len(variables_to_resotre))

    print(len(variables))

    # a = darknet53(inputs, True)
    anchors = np.array([[1, 2], [3, 4], [5, 6]])
    # grid, feats, box_xy, box_wh = yolov3_head(a, anchors, 20, (416, 416), True)

    #b = tf.global_variables(scope='yolov3')
    #print(tf.global_variables())
    #print(len(tf.global_variables()))

    #params = 0
    #for i in range(len(tf.global_variables())):
    #    # if 'conv' not in b[i].name.split('/')[-2] and 'batch_normalization' not in b[i].name.split('/')[-2]:
    #    print(b[i].name, b[i].shape)

    #    shape = b[i].shape.as_list()
    #    params += np.prod(shape)

    #print(params)

    # print(a)
    # print(b)
    # print(c)

    # grid, feats, box_xy, box_wh = yolov3_head(inputs, anchors, 20, (416, 416), True)
    # yolo_outputs = [tf.constant(0.5, shape=[4, 13, 13, 75]), tf.constant(0.5, shape=[4, 26, 26, 75]), tf.constant(0.5, shape=[4, 52, 52, 75])]

    # y_true = [tf.constant(0.5, shape=[4, 13, 13, 3, 25]), tf.constant(0.5, shape=[4, 26, 26, 3, 25]), tf.constant(0.5, shape=[4, 52, 52, 3, 25])]

    # loss = yolov3_loss(yolo_outputs=yolo_outputs, y_true=y_true, num_classes=20)

    # print(loss)

    # with tf.Session() as sess:
    # yolo_outputs = [tf.constant(0.5, shape=[4, 13, 13, 75]), tf.constant(0.5, shape=[4, 26, 26, 75]),
    # tf.constant(0.5, shape=[4, 52, 52, 75])]

    # boxes_, scores_, classes_ = yolo_eval(yolo_outputs, num_classes=20, image_shape=(500, 400), max_boxes=30, score_threshold=.6, iou_threshold=.5)


if __name__ == '__main__':
    ignore_test()