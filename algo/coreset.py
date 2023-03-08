#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
from torch.utils.data import Dataset
import os.path as osp
import os
import numpy as np

# Mini batch manager
class DatasetBuffer(Dataset):
    def __init__(self, buffer_size, class_num):
        self.buffer_size = buffer_size
        self.class_num = class_num
        self.c_n = list()
        self.c_r = list()

    def append(self, ipt, opt):
        if len(self.c_n) >= self.buffer_size:
            self.c_n.pop()
            self.c_r.pop()

        ipt_post = np.array([ipt])
        opt_post = float(opt)

        self.c_n.append(ipt_post)
        self.c_r.append(opt_post)

    def __getitem__(self, idx):
        return [self.c_n[idx], self.c_r[idx]]
    
    def __len__(self):
        return len(self.c_n)
    