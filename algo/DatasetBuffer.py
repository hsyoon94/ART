#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
from torch.utils.data import Dataset
import os.path as osp
import os
import numpy as np

# Mini batch manager
class DatasetBuffer(Dataset):
    def __init__(self, buffer_size, buffer_type, class_num, mode):
        self.buffer_size = buffer_size
        self.buffer_type = buffer_type
        self.class_num = class_num
        self.class_num_array = np.zeros((self.class_num, 0))
        self.mode = mode
        self.c_n = list()
        self.c_r = list()
        
        self.c_r_mild = list()
        self.c_r_moderate = list()
        self.c_r_harsh = list()

    def append(self, ipt, opt):
        
        if self.mode == "train":
            if len(self.c_n) >= self.buffer_size:
                self.c_n.pop()
                self.c_r.pop()

            ipt_post = np.array([ipt])
            opt_post = opt

            self.c_n.append(ipt_post)
            self.c_r.append(opt_post)
        
        elif self.mode == "inference":
            ipt_post = np.array([ipt])
            self.c_n.append(ipt_post)

    def __getitem__(self, idx):
        if self.mode == "train":
            return [self.c_n[idx], self.c_r[idx]]
        elif self.mode == "inference":
            return self.c_n[idx]
    
    def __len__(self):
        return len(self.c_n)
    
