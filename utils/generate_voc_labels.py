#coding=utf-8

from __future__ import print_function
from __future__ import division

import os
import numpy as np
import argparse
import xml.etree.ElementTree as ET

def generate_voc_labels(image_path, xml_path, filenames_list_path, data_type, saved_path):

    classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
               'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
               'train', 'tvmonitor']

    f = open(filenames_list_path, 'r')
    filenames_list = f.readlines()

    if not os.path.exists(saved_path):
        os.mkdir(saved_path)

    f_save = open(os.path.join(saved_path, data_type+'.txt'), 'w')


    for filename in filenames_list:
        label = ''
        image_file_path = os.path.join(image_path, filename.strip() + '.jpg')

        label += image_file_path


        xml_file_path = os.path.join(xml_path, filename.strip() + '.xml')
        #print(image_file_path, xml_file_path)

        root = ET.parse(xml_file_path).getroot()
        objects = root.findall('object')

        for obj in objects:

            bbox = obj.find('bndbox')
            xmin = bbox.find('xmin').text.strip()
            xmax = bbox.find('xmax').text.strip()
            ymin = bbox.find('ymin').text.strip()
            ymax = bbox.find('ymax').text.strip()


            class_ind = classes.index(obj.find('name').text.lower().strip())
            #print(class_ind)
            #print(','.join([xmin, ymin, xmax, ymax, str(class_ind)]))
            label += ' ' + ','.join([xmin, ymin, xmax, ymax, str(class_ind)])

        f_save.write(label + '\n')

    print('%s set is ok' % data_type)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default='/Users/zhulf/data/VOCdevkit/VOC2012/JPEGImages', help='the path to the raw images')
    parser.add_argument('--xml_path', type=str, default='/Users/zhulf/data/VOCdevkit/VOC2012/Annotations', help='the path to the xml files')

    parser.add_argument('--train_filenames_list_path', type=str, default='/Users/zhulf/data/VOCdevkit/VOC2012/ImageSets/Main/train.txt', help='the path to the train filenames list')
    parser.add_argument('--val_filenames_list_path', type=str, default='/Users/zhulf/data/VOCdevkit/VOC2012/ImageSets/Main/val.txt', help='the path to the val filenames list')
    parser.add_argument('--trainval_filenames_list_path', type=str, default='/Users/zhulf/data/VOCdevkit/VOC2012/ImageSets/Main/trainval.txt', help='the path to the trainval filenames list')

    parser.add_argument('--saved_path', type=str, default='../data', help='the path to save label files')

    args = parser.parse_args()

    image_path = args.image_path
    xml_path = args.xml_path
    train_filenames_list_path = args.train_filenames_list_path
    val_filenames_list_path = args.val_filenames_list_path
    trainval_filenames_list_path = args.trainval_filenames_list_path
    saved_path = args.saved_path

    generate_voc_labels(image_path, xml_path, train_filenames_list_path, 'train', saved_path)

    generate_voc_labels(image_path, xml_path, val_filenames_list_path, 'val', saved_path)

    generate_voc_labels(image_path, xml_path, trainval_filenames_list_path, 'trainval', saved_path)

