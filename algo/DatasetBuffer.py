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
if torch.cuda.device_count() > 1:
    device = torch.device('cuda:2' if is_cuda else 'cpu')
else:
    device = torch.device('cuda:0' if is_cuda else 'cpu')

# Mini batch manager
class ARTBuffer(Dataset):
    def __init__(self, buffer_size, buffer_type, class_num, dataset_type, mode):
        self.buffer_type = buffer_type
        self.class_num = class_num
        self.buffer_size = int(buffer_size / self.class_num)
        self.mode = mode
        self.c_n = list()
        self.c_r = list()
        self.softmax = nn.Softmax(dim=1)
        self.dataset_type = dataset_type
        self.coreset_management_start = False
        
        self.c_n_classwise = list()
        self.c_r_classwise = list()
        self.class_count_full = np.zeros(self.class_num)

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
                # if isinstance(opt, np.ndarray):
                #     opt_idx = int(opt[0])
                # else:
                #     opt_idx = opt
                print("new ipt", ipt)
                print("new opt", opt)

                for pair in zip(ipt, opt):
                    ipt_post = pair[0]
                    opt_idx = pair[1]
                    
                    # Input data process
                    if self.dataset_type == 'husky':
                        ipt_post = np.array([ipt_post])
                    
                    opt_post = opt_idx

                    self.class_count_full[opt_idx] = self.class_count_full[opt_idx] + 1

                    # Pop if full size
                    if len(self.c_n_classwise[opt_idx]) >= self.buffer_size:
                        self.coreset_management_start = True

                        if self.buffer_type == 'fifo':
                            self.c_n_classwise[opt_idx].pop()
                            self.c_r_classwise[opt_idx].pop()

                            self.c_n_classwise[opt_idx].append(ipt_post)
                            self.c_r_classwise[opt_idx].append(opt_idx)

                        elif self.buffer_type == 'random':
                            rnd_idx = np.random.randint(self.buffer_size, size=1)
                            self.c_n_classwise[opt_idx].pop(rnd_idx[0])
                            self.c_r_classwise[opt_idx].pop(rnd_idx[0])

                            self.c_n_classwise[opt_idx].append(ipt_post)
                            self.c_r_classwise[opt_idx].append(opt_idx)
                        
                        elif self.buffer_type == 'ucm':
                            with torch.no_grad():
                                tmp_input = torch.stack(self.c_n_classwise[opt_idx], dim=0).type(torch.FloatTensor).to(device)
                                output_softmax_list = list()
                                
                                for ensemble_idx in range(ensemble_cycle):
                                    
                                    output = model(tmp_input).squeeze()
                                    output_softmax = self.softmax(output).detach()
                                    output_softmax_list.append(output_softmax)

                                output_softmax_history = torch.stack(output_softmax_list, dim=0).to(device)
                                output_std = torch.std(output_softmax_history, 0, True).squeeze()
                                
                                output_std_sum = torch.sum(output_std, 1)
                                
                                output_std_argmin_idx = torch.argmin(output_std_sum)
                                
                                self.c_n_classwise[opt_idx].pop(output_std_argmin_idx)
                                self.c_r_classwise[opt_idx].pop(output_std_argmin_idx)
                                
                                self.c_n_classwise[opt_idx].append(ipt_post)
                                self.c_r_classwise[opt_idx].append(opt_idx)

                        elif self.buffer_type == 'reservoir':
                            m_c = len(self.c_n_classwise[opt_idx])
                            n_c = self.class_count_full[opt_idx]
                            random_probability = random.uniform(0, 1)
                            if random_probability <= (m_c / n_c):
                                random_idx = random.sample(range(0, len(self.c_n_classwise[opt_idx])), 1)
                                for tmp_idx in range(1):
                                    self.c_n_classwise[opt_idx][random_idx[tmp_idx]] = ipt_post
                                    self.c_r_classwise[opt_idx][random_idx[tmp_idx]] = opt_idx
                        
                    else:
                        self.c_n_classwise[opt_idx].append(ipt_post)
                        self.c_r_classwise[opt_idx].append(opt_idx)
                    
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


    def initial_append(self, ipt, opt, model, ensemble_cycle):
        if self.mode == "train":
            
            if self.class_num != 1:
                # if isinstance(opt, np.ndarray):
                #     opt_idx = int(opt[0])
                # else:
                #     opt_idx = opt
                
                ipt_post = ipt
                opt_idx = opt
                    
                # Input data process
                if self.dataset_type == 'husky':
                    ipt_post = np.array([ipt_post])
                
                opt_post = opt_idx

                self.class_count_full[opt_idx] = self.class_count_full[opt_idx] + 1

                # Pop if full size
                if len(self.c_n_classwise[opt_idx]) >= self.buffer_size:
                    self.coreset_management_start = True

                    if self.buffer_type == 'fifo':
                        self.c_n_classwise[opt_idx].pop()
                        self.c_r_classwise[opt_idx].pop()

                        self.c_n_classwise[opt_idx].append(ipt_post)
                        self.c_r_classwise[opt_idx].append(opt_idx)

                    elif self.buffer_type == 'random':
                        rnd_idx = np.random.randint(self.buffer_size, size=1)
                        self.c_n_classwise[opt_idx].pop(rnd_idx[0])
                        self.c_r_classwise[opt_idx].pop(rnd_idx[0])

                        self.c_n_classwise[opt_idx].append(ipt_post)
                        self.c_r_classwise[opt_idx].append(opt_idx)
                    
                    elif self.buffer_type == 'ucm':
                        with torch.no_grad():
                            tmp_input = torch.stack(self.c_n_classwise[opt_idx], dim=0).type(torch.FloatTensor).to(device)
                            output_softmax_list = list()
                            
                            for ensemble_idx in range(ensemble_cycle):
                                
                                output = model(tmp_input).squeeze()
                                output_softmax = self.softmax(output).detach()
                                output_softmax_list.append(output_softmax)

                            output_softmax_history = torch.stack(output_softmax_list, dim=0).to(device)
                            output_std = torch.std(output_softmax_history, 0, True).squeeze()
                            
                            output_std_sum = torch.sum(output_std, 1)
                            
                            output_std_argmin_idx = torch.argmin(output_std_sum)
                            
                            self.c_n_classwise[opt_idx].pop(output_std_argmin_idx)
                            self.c_r_classwise[opt_idx].pop(output_std_argmin_idx)
                            
                            self.c_n_classwise[opt_idx].append(ipt_post)
                            self.c_r_classwise[opt_idx].append(opt_idx)

                    elif self.buffer_type == 'reservoir':
                        m_c = len(self.c_n_classwise[opt_idx])
                        n_c = self.class_count_full[opt_idx]
                        random_probability = random.uniform(0, 1)
                        if random_probability <= (m_c / n_c):
                            random_idx = random.sample(range(0, len(self.c_n_classwise[opt_idx])), 1)
                            for tmp_idx in range(1):
                                self.c_n_classwise[opt_idx][random_idx[tmp_idx]] = ipt_post
                                self.c_r_classwise[opt_idx][random_idx[tmp_idx]] = opt_idx
                    
                else:
                    self.c_n_classwise[opt_idx].append(ipt_post)
                    self.c_r_classwise[opt_idx].append(opt_idx)
                
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


class CBRSBuffer(Dataset):
    def __init__(self, buffer_size, class_num, dataset_type):
        
        self.buffer_type = buffer_type
        self.class_num = class_num
        self.buffer_size = buffer_size
        self.c_n = list()
        self.c_r = list()
        self.dataset_type = dataset_type
        
        self.c_n_classwise = list()
        self.c_r_classwise = list()
        self.class_count_full = np.zeros(self.class_num)

        for i in range(self.class_num):
            self.c_n_classwise.append(list())
            self.c_r_classwise.append(list())

    def append(self, ipt, opt):
        if self.mode == "train":
            
            if self.class_num != 1:
                if isinstance(opt, np.ndarray):
                    opt_idx = int(opt[0])
                else:
                    opt_idx = opt
                
                # Input data process
                if self.dataset_type == 'husky':
                    ipt_post = np.array([ipt])
                else:
                    ipt_post = ipt
                opt_post = opt
                
                self.class_count_full[opt_idx] = self.class_count_full[opt_idx] + 1

                # Pop if full size
                if len(self.c_n) >= self.buffer_size:
                    # Select largest class
                    max_class_idx = np.argmax(self.class_count_full)

                    if max_class_idx != opt_post: # Line 6
                        random_idx = random.sample(range(0, self.c_n_classwise[max_class_idx]), ipt_post.size()[0]) # Line 7, 8
                        for rnd_idx in random_idx: # Line 9
                            self.c_n_classwise[max_class_idx].pop(rnd_idx)
                            self.c_r_classwise[max_class_idx].pop(rnd_idx)
                        self.c_n_classwise[opt_idx].append(ipt_post)
                        self.c_r_classwise[opt_idx].append(opt_post)
                    else: # Line 10
                        m_c = len(self.c_n_classwise[opt_idx]) # Line 11
                        n_c = self.class_count_full[opt_idx] # Line 12
                        random_probability = random.uniform(0, 1) # Line 13
                        if random_probability <= m_c / n_c: # Line 14
                            random_idx = random.sample(range(0, self.c_n_classwise[opt_idx]), ipt_post.size()[0]) # Line 15
                            for tmp_idx in range(ipt_post.size()[0]): # Line 16
                                self.c_n_classwise[opt_idx][random_idx[tmp_idx]] = ipt_post[tmp_idx]
                                self.c_r_classwise[opt_idx][random_idx[tmp_idx]] = opt_post[tmp_idx]

                else:
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