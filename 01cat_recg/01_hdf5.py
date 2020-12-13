#!/usr/bin/env python
# -*- coding:utf-8 -*-
# datetime:2020/2/20 17:16

import numpy as np
import matplotlib.pyplot as plt  # painting
import h5py  # data management
import skimage.transform as tf  # zoom image


with h5py.File('datasets/train_catvnoncat.h5', "r") as f:
    for key in f.keys():
        print(f[key], key, f[key].name)

# https://blog.csdn.net/buchidanhuang/article/details/89716252?utm_source=distribute.pc_relevant.none-task
