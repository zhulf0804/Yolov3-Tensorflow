#coding=utf-8

from __future__ import print_function
from __future__ import division


import tensorflow as tf
import math
import numpy as np
from utils.get_anchors import get_anchors

#Used for BN
_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


def weight_variable(shape, stddev=None, name='weight'):
    if stddev == None:
        if len(shape) == 4:
            stddev = math.sqrt(2. / (shape[0] * shape[1] * shape[2]))
        else:
            stddev = math.sqrt(2. / shape[0])
    else:
        stddev = 0.1
    initial = tf.truncated_normal(shape, stddev=stddev)
    W = tf.Variable(initial, name=name)

    return W


def bias_variable(shape, name='bias'):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def batch_norm(inputs, is_training):

  return tf.layers.batch_normalization(
      inputs=inputs, axis=-1,
      momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
      scale=True, training=is_training, fused=True)

def convolutional(inputs, weight, stride, is_training, length=-1, relu=True, bn=True):
    net = tf.nn.conv2d(inputs, weight, [1, stride, stride, 1], padding='SAME')
    if bn:
        net = batch_norm(net, is_training=is_training)
    else:

        bias =  bias_variable([length], name='bias')
        net = tf.nn.bias_add(net, bias)

    if relu:
        net = tf.nn.leaky_relu(net, alpha=0.1)

    return net

def residual(inputs, num_featues, is_training):

    shortcut = inputs
    weight_1 = weight_variable([1, 1, num_featues, num_featues // 2], name='weight_1')
    net = convolutional(inputs, weight_1, 1, is_training)
    weight_3 = weight_variable([3, 3, num_featues // 2, num_featues], name='weight_3')
    net = convolutional(net, weight_3, 1, is_training)

    return shortcut + net

def darknet53(inputs, is_training):

    with tf.name_scope('init'):
        weight_3_1 = weight_variable([3, 3, 3, 32], name='weight_3_1')
        net = convolutional(inputs, weight_3_1, 1, is_training)

        weight_3_2 = weight_variable([3, 3, 32, 64], name='weight_3_2')
        net = convolutional(net, weight_3_2, 2, is_training)

    with tf.name_scope('convs_1'):
        for i in range(1):
            with tf.name_scope("conv_%d"%i):
                net = residual(net, 64, is_training)

        weight_3 = weight_variable([3, 3, 64, 128])
        net = convolutional(net, weight_3, 2, is_training)

    with tf.name_scope('convs_2'):
        for i in range(2):
            with tf.name_scope('conv_%d'%i):
                net = residual(net, 128, is_training)

        weight_3 = weight_variable([3, 3, 128, 256])
        net = convolutional(net, weight_3, 2, is_training)

    with tf.name_scope('convs_3'):
        for i in range(8):
            with tf.name_scope('conv_%d'%i):
                net = residual(net, 256, is_training)

        route_1 = net

        weight_3 = weight_variable([3, 3, 256, 512])
        net = convolutional(net, weight_3, 2, is_training)

    with tf.name_scope('convs_4'):
        for i in range(8):
            with tf.name_scope('conv_%d'%i):
                net = residual(net, 512, is_training)

        route_2 = net


        weight_3 = weight_variable([3, 3, 512, 1024])
        net = convolutional(net, weight_3, 2, is_training)

    with tf.name_scope('convs_5'):
        for i in range(2):
            with tf.name_scope('conv_%d'%i):
                net = residual(net, 1024, is_training)

    return route_1, route_2, net

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
    box_xy = tf.nn.sigmoid(feats[..., :2] + grid) / tf.cast(grid_shape[::-1], tf.float32)
    box_wh = tf.exp(feats[..., 2:4]) * tf.cast(anchors_tensor, tf.float32) / tf.cast(input_shape[::-1], tf.float32)
    box_cofidence = tf.nn.sigmoid(feats[..., 4:5])
    box_class_probs = tf.nn.sigmoid(feats[..., 5:])

    if calc_loss:
        return grid, feats, box_xy, box_wh

    return box_xy, box_wh, box_cofidence, box_class_probs

def decode(conv_output, anchors, stride, classes):
    '''
    return tensors of shape [batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes]

    :param conv_output:
    :param anchors:
    :param stride:
    :return:
    '''

    conv_shape = tf.shape(conv_output)
    batch_size = conv_shape[0]
    output_size = conv_shape[1]
    anchor_per_scale = len(anchors)

    conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, anchor_per_scale, 5 + classes))

    conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
    conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
    conv_raw_conf = conv_output[:, :, :, :, 4:5]
    conv_raw_prob = conv_output[:, :, :, :, 5:]


    y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
    x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])

    xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
    xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, anchor_per_scale, 1])
    xy_grid = tf.cast(xy_grid, tf.float32)

    pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * stride
    pred_wh = (tf.exp(conv_raw_dwdh) * anchors) * stride
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_prob = tf.sigmoid(conv_raw_prob)

    return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

def yolov3_body(inputs, is_training, classes):

    anchors = get_anchors()
    strides = np.array([8, 16, 32])


    with tf.name_scope("darknet53"):
        route_1, route_2, net = darknet53(inputs, is_training=is_training)

    with tf.name_scope("large_obj"):
        weight_1_1 = weight_variable([1, 1, 1024, 512], name='weight_1_1')
        net = convolutional(net, weight_1_1, 1, is_training)

        weight_3_1 = weight_variable([3, 3, 512, 1024], name='weight_3_1')
        net = convolutional(net, weight_3_1, 1, is_training)

        weight_1_2 = weight_variable([1, 1, 1024, 512], name='weight_1_2')
        net = convolutional(net, weight_1_2, 1, is_training)

        weight_3_2 = weight_variable([3, 3, 512, 1024], name='weight_3_2')
        net = convolutional(net, weight_3_2, 1, is_training)

        weight_1_3 = weight_variable([1, 1, 1024, 512], name='weight_1_3')
        net = convolutional(net, weight_1_3, 1, is_training)


        weight_3 = weight_variable([3, 3, 512, 1024], name='weight_3')
        conv_lobj_branch = convolutional(net, weight_3, 1, is_training)

        weight_1 = weight_variable([1, 1, 1024, 3*(classes + 5)], name='weight_1')
        y1 = convolutional(conv_lobj_branch, weight_1, 1, is_training, length=3*(classes + 5), relu=False, bn=False)

    with tf.name_scope('middle_obj'):
        with tf.name_scope('up_sample'):
            weight_1 = weight_variable([1, 1, 512, 256], name='weight_1')
            net = convolutional(net, weight_1, 1, is_training)
            net = tf.image.resize_nearest_neighbor(net, tf.shape(net)[1:3]*2)

        net = tf.concat([net, route_2], axis=-1)

        weight_1_1 = weight_variable([1, 1, 768, 256], name='weight_1_1')
        net = convolutional(net, weight_1_1, 1, is_training)

        weight_3_1 = weight_variable([3, 3, 256, 512], name='weight_3_1')
        net = convolutional(net, weight_3_1, 1, is_training)

        weight_1_2 = weight_variable([1, 1, 512, 256], name='weight_1_2')
        net = convolutional(net, weight_1_2, 1, is_training)

        weight_3_2 = weight_variable([3, 3, 256, 512], name='weight_3_2')
        net = convolutional(net, weight_3_2, 1, is_training)

        weight_1_3 = weight_variable([1, 1, 512, 256], name='weight_1_3')
        net = convolutional(net, weight_1_3, 1, is_training)

        weight_3 = weight_variable([3, 3, 256, 512], name='weight_3')
        conv_mobj_branch = convolutional(net, weight_3, 1, is_training)

        weight_1 = weight_variable([1, 1, 512, 3 * (classes + 5)], name='weight_1')
        y2 = convolutional(conv_mobj_branch, weight_1, 1, is_training, length=3 * (classes + 5), relu=False,
                                   bn=False)

    with tf.name_scope('small_obj'):
        with tf.name_scope('up_sample'):
            weight_1 = weight_variable([1, 1, 256, 128], name='weight_1')
            net = convolutional(net, weight_1, 1, is_training)
            net = tf.image.resize_nearest_neighbor(net, tf.shape(net)[1:3]*2)

        net = tf.concat([net, route_1], axis=-1)

        weight_1_1 = weight_variable([1, 1, 384, 128], name='weight_1_1')
        net = convolutional(net, weight_1_1, 1, is_training)

        weight_3_1 = weight_variable([3, 3, 128, 256], name='weight_3_1')
        net = convolutional(net, weight_3_1, 1, is_training)

        weight_1_2 = weight_variable([1, 1, 256, 128], name='weight_1_2')
        net = convolutional(net, weight_1_2, 1, is_training)

        weight_3_2 = weight_variable([3, 3, 128, 256], name='weight_3_2')
        net = convolutional(net, weight_3_2, 1, is_training)

        weight_1_3 = weight_variable([1, 1, 256, 128], name='weight_1_3')
        net = convolutional(net, weight_1_3, 1, is_training)

        weight_3 = weight_variable([3, 3, 128, 256], name='weight_3')
        conv_sobj_branch = convolutional(net, weight_3, 1, is_training)

        weight_1 = weight_variable([1, 1, 256, 3 * (classes + 5)], name='weight_1')
        y3 = convolutional(conv_sobj_branch, weight_1, 1, is_training, length=3 * (classes + 5), relu=False,
                                   bn=False)

    #with tf.name_scope('decode'):
    #    pred_sbbox = decode(conv_sbbox, anchors[0], strides[0], classes)
    #    pred_mbbox = decode(conv_mbbox, anchors[1], strides[1], classes)
    #    pred_lbbox = decode(conv_lbbox, anchors[2], strides[2], classes)


    return y1, y2, y3

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

    anchors = get_anchors()

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

        raw_true_wh = tf.math.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1]) # (batch_size, 13, 13, 3, 2) / (3, 2) * (2, 1) => (batch_size, 13, 13, 3, 2)

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


if __name__ == '__main__':
    inputs = tf.constant(0.5, shape=[4, 13, 13, 75])

    #a, b, c = yolov3(inputs, is_training=True, classes=20)

    #print(a)
    #print(b)
    #print(c)

    #grid, feats, box_xy, box_wh = yolov3_head(inputs, anchors, 20, (416, 416), True)
    yolo_outputs = [tf.constant(0.5, shape=[4, 13, 13, 75]), tf.constant(0.5, shape=[4, 26, 26, 75]), tf.constant(0.5, shape=[4, 52, 52, 75])]




    y_true = [tf.constant(0.5, shape=[4, 13, 13, 3, 25]), tf.constant(0.5, shape=[4, 26, 26, 3, 25]), tf.constant(0.5, shape=[4, 52, 52, 3, 25])]

    #loss = yolov3_loss(yolo_outputs=yolo_outputs, y_true=y_true, num_classes=20)



    #print(loss)

    boxes_, scores_, classes_ = yolo_eval(yolo_outputs, num_classes=20, image_shape=(500, 400), max_boxes=30, score_threshold=.6, iou_threshold=.5)

    print(boxes_.shape)
