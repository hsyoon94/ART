#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn

from torch.utils.data import Dataset
import os.path as osp
import os
import numpy as np
import random

is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')

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
        self.softmax = nn.Softmax(dim=1)

        
        self.c_n_classwise = list()
        self.c_r_classwise = list()

        for i in range(self.class_num):
            self.c_n_classwise.append(list())
            self.c_r_classwise.append(list())

    def sequantial_append(self, ipt, opt):
        
        if self.mode == "train": 
            if len(self.c_n) >= self.buffer_size:
                self.c_n.pop()
                self.c_r.pop()

            ipt_post = np.array([ipt])
            opt_post = opt

            self.c_n.append(ipt_post)
            self.c_r.append(opt_post)

    def append(self, ipt, opt, model, ensemble_cycle):
        if self.mode == "train":
            
            if self.class_num != 1:
                opt_idx = np.argmax(opt)
                if len(self.c_n_classwise[opt_idx]) >= self.buffer_size:
                    if self.buffer_type == 'fifo':
                        self.c_n_classwise[opt_idx].pop()
                        self.c_r_classwise[opt_idx].pop()
                    
                    elif self.buffer_type == 'random':
                        rnd_idx = np.random.randint(self.buffer_size, size=1)
                        self.c_n_classwise[opt_idx].pop(rnd_idx[0])
                        self.c_r_classwise[opt_idx].pop(rnd_idx[0])
                    
                    elif self.buffer_type == 'ucm':
                        with torch.no_grad():
                            tmp_input = torch.tensor(self.c_n_classwise[opt_idx]).squeeze().type(torch.FloatTensor).to(device)                            
                            output_softmax_history = torch.empty((ensemble_cycle, len(self.c_n_classwise[opt_idx]), self.class_num))

                            for ensemble_idx in range(ensemble_cycle):
                                output = model(tmp_input).squeeze()
                                output_softmax = self.softmax(output).detach()
                                output_softmax_history[ensemble_idx, :, :] = output_softmax
                                
                            output_std = torch.std(output_softmax_history, 0, True).squeeze()
                            output_std_sum = torch.sum(output_std, 1)
                            output_std_argmin_idx = torch.argmin(output_std_sum)
                            
                            self.c_n_classwise[opt_idx].pop(output_std_argmin_idx)
                            self.c_r_classwise[opt_idx].pop(output_std_argmin_idx)


                ipt_post = np.array([ipt])
                opt_post = opt

                self.c_n_classwise[opt_idx].append(ipt_post)
                self.c_r_classwise[opt_idx].append(opt_post)
                cur_buffer_length = 0

                for i in range(self.class_num):
                    cur_buffer_length = cur_buffer_length + len(self.c_n_classwise[i])

                random_index = random.sample(range(0, cur_buffer_length), cur_buffer_length)
                
                self.c_n = self._list_flatten(self.c_n_classwise)
                self.c_r = self._list_flatten(self.c_r_classwise)

                self.c_n = [self.c_n[i] for i in random_index]
                self.c_r = [self.c_r[j] for j in random_index]

            elif self.class_num == 1:
                if self.buffer_type == 'fifo':
                    if len(self.c_n) >= self.buffer_size:
                        self.c_n.pop()
                        self.c_r.pop()

                    self.c_n.append(ipt)
                    self.c_r.append(opt)

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

    def get_labels(self):
        return self.c_r
    
    def _list_flatten(self, lst):
        result = []
        for item in lst:
            if type(item) == list:
                result = result + self._list_flatten(item)
            else:
                result = result + [item]
    
        return result