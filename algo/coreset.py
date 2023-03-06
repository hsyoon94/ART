#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
from torch.utils.data import Dataset
import os.path as osp
import os
import numpy as np


# Mini batch manager
class CoresetBuffer(Dataset):
    def __init__(self, buffer_size, class_num):
        self.buffer_size = buffer_size
        self.class_num = class_num
        

    def append(self, ipt, opt):
