#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
from torch.utils.data import Dataset
import os.path as osp
import os
import numpy as np
import random

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
        
        self.c_n_mild = list()
        self.c_n_moderate = list()
        self.c_n_harsh = list()
        self.c_r_mild = list()
        self.c_r_moderate = list()
        self.c_r_harsh = list()

    def sequantial_append(self, ipt, opt):
        
        if self.mode == "train": 
            if len(self.c_n) >= self.buffer_size:
                self.c_n.pop()
                self.c_r.pop()

            ipt_post = np.array([ipt])
            opt_post = opt

            self.c_n.append(ipt_post)
            self.c_r.append(opt_post)

    def append(self, ipt, opt):
        
        if self.mode == "train":
            
            if opt[0] == 1:
                if len(self.c_n_mild) >= self.buffer_size:
                    self.c_n_mild.pop()
                    self.c_r_mild.pop()

                ipt_post = np.array([ipt])
                opt_post = opt

                self.c_n_mild.append(ipt_post)
                self.c_r_mild.append(opt_post)

            elif opt[1] == 1:
                if len(self.c_n_moderate) >= self.buffer_size:
                    self.c_n_moderate.pop()
                    self.c_r_moderate.pop()

                ipt_post = np.array([ipt])
                opt_post = opt

                self.c_n_moderate.append(ipt_post)
                self.c_r_moderate.append(opt_post)

            elif opt[2] == 1:
                if len(self.c_n_harsh) >= self.buffer_size:
                    self.c_n_harsh.pop()
                    self.c_r_harsh.pop()

                ipt_post = np.array([ipt])
                opt_post = opt

                self.c_n_harsh.append(ipt_post)
                self.c_r_harsh.append(opt_post)

            cur_buffer_length = len(self.c_n_mild) + len(self.c_n_moderate) + len(self.c_n_harsh)
            random_index = random.sample(range(0, cur_buffer_length), cur_buffer_length)
            
            self.c_n = self.c_n_mild + self.c_n_moderate + self.c_n_harsh
            self.c_r = self.c_r_mild + self.c_r_moderate + self.c_r_harsh
            
            self.c_n = [self.c_n[i] for i in random_index]
            self.c_r = [self.c_r[j] for j in random_index]

            # print("c_n_mild", len(self.c_n_mild))
            # print("c_r_mild", len(self.c_r_mild))
            # print("c_n_moderate", len(self.c_n_moderate))
            # print("c_r_moderate", len(self.c_r_moderate))
            # print("c_n_harsh", len(self.c_n_harsh))
            # print("c_r_harsh", len(self.c_r_harsh))
        
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