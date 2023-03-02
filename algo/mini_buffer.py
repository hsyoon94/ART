#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
from torch.utils.data import Dataset
import os.path as osp
import os
import numpy as np




# Mini batch manager
class MiniBufferManager():
    def __init__(self, buffer_size, class_num):
        self.buffer_size = buffer_size
        self.class_num = class_num


# Training with mini batch
class TrainingWithMiniBuffer(Dataset):
    def __init__(self, rootpth, mode='train', *args, **kwargs):
        super(TrainingWithMiniBuffer, self).__init__(*args, **kwargs)

        self.input = list()
        self.output = list()
        self.len = len(self.input)

    def __getitem__(self, idx):
        
        input_ = self.input[idx]
        output_ = self.output[idx]

        return input_, output_

    def __len__(self):
        return self.len