#coding=utf-8

from __future__ import print_function
from __future__ import division

import os
import cv2
import numpy as np
import random
from utils.draw_bbox import draw_bbox_from_boxes
from utils.get_anchors import get_anchors

class Dataset(object):

    def __init__(self, dataset_type):
        self.data_aug = True if dataset_type == 'train' else False
        self.train_input_size = 416
        self.num_classes = 80
        self.anchor_per_scale = 3
        self.strides = np.array([8, 16, 32])
        self.train_output_sizes = self.train_input_size // self.strides
        self.num_samples = 5716
        self.max_bbox_per_scale = 150
        self.anchors = get_anchors()
        self.annotations = self.load_annotations(dataset_type)
        self.batch_count = 0

    def load_annotations(self, dataset_type):
        with open(os.path.join('data', dataset_type+'.txt'), 'r') as f:
            lines = f.readlines()
            annotations = [line.strip() for line in lines if len(line.strip().split()[1:]) != 0]

            np.random.shuffle(annotations)

            return annotations

    def __next__(self, batch_size, max_boxes_num=80):

        batch_images = np.zeros((batch_size, self.train_input_size, self.train_input_size, 3))
        batch_boxes = np.zeros(shape=[batch_size, max_boxes_num, 5])

        for i in range(batch_size):
            index = self.batch_count * batch_size + i
            if index > self.num_samples:
                index %= self.num_samples

            annotation = self.annotations[index]

            image, bboxes = self.parse_annotation(annotation)

            batch_images[i, ...] = image

            mmin_num = min(max_boxes_num, len(bboxes))
            batch_boxes[i, 0:mmin_num, :] = bboxes[0:mmin_num, :]


        self.batch_count += 1

        y_true = self.process_true_boxesv2(batch_boxes)

        return batch_images, y_true[0], y_true[1], y_true[2]

    def letterbox_resize(self, img, new_width, new_height, interp=0):
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

        return image_padded, resize_ratio, dw, dh

    def image_preprocess(self, image, target_size, gt_boxes=None):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        #h, w, _ = image.shape
        t_h, t_w = target_size
        image_padded, scale, dw, dh = self.letterbox_resize(image, t_w, t_h)

        #scale = min(t_h / h, t_w / w)

        #n_w, n_h = int(scale * w), int(scale * h)

        #image_resized = cv2.resize(image, (n_w, n_h))

        #image_padded = np.full(shape=[t_h, t_w, 3], fill_value=128.0)

        #dw, dh = (t_w - n_w) // 2, (t_h - n_h) // 2

        #image_padded[dh:n_h+dh, dw:n_w+dw, :] = image_resized
        image_padded = image_padded / 255.

        if gt_boxes is None:
            return image_padded

        else:
            gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
            gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh

            return image_padded, gt_boxes

    def random_horizontal_flip(self, image, bboxes):
        if random.random() < 0.5:
            _, w, _ = image.shape

            image = cv2.flip(image, 1)
            bboxes[:, [0, 2]] = w - bboxes[:, [2, 0]]

        return image, bboxes

    def random_crop(self, image, bboxes):
        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
            crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
            crop_xmax = max(w, int(max_bbox[2] + random.uniform(0, max_r_trans)))
            crop_ymax = max(h, int(max_bbox[3] + random.uniform(0, max_d_trans)))

            image = image[crop_ymin : crop_ymax, crop_xmin : crop_xmax]

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin

            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin

        return image, bboxes

    def random_translate(self, image, bboxes):
        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
            ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

            M = np.array([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, M, (w, h))

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty

        return image, bboxes

    def parse_annotation(self, annotation):

        line = annotation.split()
        image_path = line[0]

        if not os.path.exists(image_path):
            raise KeyError("%s does not exist ..." %image_path)

        image = cv2.imread(image_path)

        bboxes = np.array([list(map(int, box.split(','))) for box in line[1:]])

        if self.data_aug:
            image, bboxes = self.random_horizontal_flip(image, bboxes)
            image, bboxes = self.random_crop(image, bboxes)
            image, bboxes = self.random_translate(image, bboxes)


        image, bboxes = self.image_preprocess(image, [self.train_input_size, self.train_input_size], bboxes)

        #label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.preprocess_true_boxes(bboxes)
        #return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes

        return image, bboxes

    def bbox_iou(self, boxes1, boxes2):

        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)

        #print(boxes1)
        #print(boxes2)

        boxes1_area = boxes1[..., 2] * boxes2[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        boxes1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5, boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)

        boxes2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5, boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])
        #print(left_up)
        #print(right_down)
        inter_section = np.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area

        return inter_area / union_area


    def preprocess_true_boxes(self, bboxes):
        label = [np.zeros((self.train_output_sizes[i], self.train_output_sizes[i], self.anchor_per_scale, 5 + self.num_classes)) for i in range(3)]

        bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(3)]

        bbox_count = np.zeros((3, ))

        for bbox in bboxes:
            bbox_coor = bbox[:4]
            bbox_class_ind = bbox[4]

            onehot = np.zeros(self.num_classes, dtype=np.float32)
            onehot[bbox_class_ind] = 1.0

            uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution

            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)
            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]


            iou = []
            exist_positive = False

            for i in range(3):
                anchors_xywh = np.zeros((self.anchor_per_scale, 4))
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5

                anchors_xywh[:, 2:4] = self.anchors[i]

                iou_scale = self.bbox_iou(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
                print(iou_scale)
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)
                    label[i][yind, xind, iou_mask, :] = 0
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:] = smooth_onehot

                    bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True


            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / self.anchor_per_scale)
                best_anchor = int(best_anchor_ind % self.anchor_per_scale)


                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)

                label[best_detect][yind, xind, best_anchor, :] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot

                bbox_ind = int(bbox_count[best_detect] % self.max_bbox_per_scale)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1


        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes


    def process_true_boxesv2(self, true_boxes):


        '''
        :param true_boxes: Shape=(batch_size, max_num_boxes, 5)
        :return:
        '''

        assert (true_boxes[..., 4] < self.num_classes).all(), 'class id must be less than num_classes'
        num_layers = self.anchor_per_scale
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

        true_boxes = np.array(true_boxes, dtype=np.float32)
        input_shape = np.array([self.train_input_size, self.train_input_size], dtype=np.int32)

        boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
        boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]

        true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]   #[0, 1]
        true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]   #[0, 1]

        m = true_boxes.shape[0]

        grid_shapes = [input_shape // {0:32, 1:16, 2:8}[l] for l in range(num_layers)]
        y_true = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5 + self.num_classes), dtype=np.float32) for l in range(num_layers)]


        anchors = np.expand_dims(self.anchors, 0)

        anchors_maxes = anchors / 2.
        anchors_mins = -anchors_maxes
        valid_mask = boxes_wh[..., 0] > 0

        for b in range(m):

            wh = boxes_wh[b, valid_mask[b]]

            if len(wh) == 0: continue
            wh = np.expand_dims(wh, -2)

            box_maxes = wh / 2.
            box_mins = - box_maxes
            intersect_mins = np.maximum(box_mins, anchors_mins)
            intersect_maxes = np.minimum(box_maxes, anchors_maxes)
            intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)

            intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
            box_area = wh[..., 0] * wh[..., 1]
            anchor_area = anchors[..., 0] * anchors[..., 1]
            iou = intersect_area / (box_area + anchor_area - intersect_area)


            # Find best anchor for each true box

            best_anchor = np.argmax(iou, axis=-1)

            for t, n in enumerate(best_anchor):
                for l in range(num_layers):
                    if n in anchor_mask[l]:
                        i = np.floor(true_boxes[b, t, 0]*grid_shapes[l][1]).astype(np.int32)
                        j = np.floor(true_boxes[b, t, 1]*grid_shapes[l][0]).astype(np.int32)

                        #print(true_boxes[b, t, 0], true_boxes[b, t, 1])
                        k = anchor_mask[l].index(n)
                        c = true_boxes[b, t, 4].astype(np.int32)
                        #print(b, j, i, k)
                        y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                        y_true[l][b, j, i, k, 4] = 1
                        y_true[l][b, j, i, k, 5 + c] = 1

        return y_true


if __name__ == '__main__':
    voc = Dataset('train')

    # test parse_annotation()

    #image, bboxes = voc.parse_annotation(',14 206,176,256,236,14 122,162,198,330,14 457,212,500,297,14 438,173,500,243,14 351,272,452,375,8 265,250,353,364,8 189,260,286,375,8 149,224,402,331,10 54,189,84,214,15 358,188,469,359,14')
    #image = draw_bbox_from_boxes(image, bboxes)
    #cv2.imwrite('1.jpg', image)

    # test iou

    #boxes1 = [[1, 1, 2, 2]]
    #boxes2 = [[1, 1, 2, 2],
    #          [2, 2, 2, 2],
    #          [3, 3, 2, 2]]

    #a = voc.bbox_iou(boxes1, boxes2)
    #print(a)

    # test preprocess_true_boxes()

    boxes = np.array([[[120,1,203,35,1],
                      [117,38,273,121,1],
                      [206,74,395,237,14],
                      [24,2,400,188,3],
                      [1,187,400,282,3]],
                      [[120, 1, 203, 35, 1],
                        [117, 38, 273, 121, 1],
                        [206, 74, 395, 237, 14],
                        [24, 2, 400, 188, 3],
                        [1, 187, 400, 282, 3]]
                        ]
                     )

    y_true = voc.process_true_boxesv2(boxes)

    print(y_true[0].shape)

    #label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = voc.preprocess_true_boxes(boxes)

    #print(label_lbbox)
    #print(label_lbbox.shape)
    #print(label_mbbox.shape)
    #print(label_sbbox.shape)
    #print(sbboxes.shape)
    #print(mbboxes.shape)
    #print(lbboxes.shape)


    '''
    # test __next__()

    batch_label_sbbox, batch_label_mbbox, batch_label_lbbox, batch_sbboxes, batch_mbboxes, batch_lbboxes = voc.parse_annotation('/Volumes/Samsung_T5/datasets/VOCdevkit/VOCdevkit/VOC2012/JPEGImages/2011_003081.jpg 287,113,455,220,3 331,138,355,179,14 245,149,276,203,14')


    print(batch_label_sbbox.shape)
    print(batch_label_mbbox.shape)
    print(batch_label_lbbox.shape)
    print(batch_sbboxes.shape)
    print(batch_mbboxes.shape)
    print(batch_lbboxes.shape)

    '''



