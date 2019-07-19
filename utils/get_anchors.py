#coding=utf-8

from __future__ import print_function
from __future__ import division

import numpy as np
import config

def get_anchors():
    anchors_path = config._anchor_txt
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = np.array(anchors.split(','), dtype=np.float32)
    return anchors.reshape(-1, 2)

if __name__ == '__main__':
    print(get_anchors())