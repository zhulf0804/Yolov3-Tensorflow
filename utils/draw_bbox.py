#coding=utf-8

from __future__ import print_function
from __future__ import division

import cv2
import os
import numpy as np


def draw_bbox(file_path):

    colormap = [[128, 0, 0], [0, 128, 0], [128, 128, 0],
                    [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                    [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                    [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                    [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                    [0, 64, 128]]

    classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
               'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
               'train', 'tvmonitor']

    f = open(file_path)
    line = f.readline().strip().split()
    image_path = line[0].strip()

    image = cv2.imread(image_path)

    for bbox in line[1:]:
        bbox = bbox.split(',')
        label = int(bbox[-1])
        image = cv2.rectangle(image, (int(float(bbox[0])), int(float(bbox[1]))), (int(float(bbox[2])), int(float(bbox[3]))), colormap[label], 2)

        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(image, classes[label], (int(float(bbox[0])), int(float(bbox[1]))), font, 0.3, (255, 255, 255), 1)

    base_name = os.path.basename(image_path)
    if not os.path.exists('output'):
        os.mkdir('output')

    cv2.imwrite(os.path.join('output', base_name), image)

    print("file saved in %s" %(os.path.join('output', base_name)))


def draw_bbox_from_boxes(image, boxes):
    for box in boxes:
        print(box)
        image = cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 255), 2)

    return image
if __name__ == '__main__':
    draw_bbox('label.txt')



