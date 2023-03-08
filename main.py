import numpy as np
import torch
from algo.ART import ART
from algo.coreset import DatasetBuffer
from dataparser.dataparser import raw_data_parser
from arguments import get_args
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os

import torch.nn as nn
import torch.optim as optim

import csv

is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')

args = get_args()

def main():
    
    model = ART(args.c_n_grid_size, args.c_n_grid_channel, args.c_r_class_num, device)
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), args.lr)
    writer = SummaryWriter()

    # Get full rosbag dataset for offline operation
    full_rosbag_dataset = raw_data_parser(args)

    # Coreset Initialization
    coreset = DatasetBuffer(args.coreset_buffer_size, args.c_r_class_num)
    for i in range(args.coreset_buffer_size):
        coreset.append(full_rosbag_dataset[i][0], full_rosbag_dataset[i][1])

    full_data_index = args.coreset_buffer_size

    for cycle_idx in range(args.training_cycle):
        print("Training cycle", cycle_idx, "/", args.training_cycle, "starts")
        
        dl = DataLoader(coreset, batch_size=2, shuffle=True, sampler=None, num_workers=0, pin_memory=False, drop_last=True)
        diter = iter(dl)

        train(args, model, diter, dl, args.iteration, cycle_idx, loss_function, optimizer, writer, device)
        writer.flush()

        new_data_ipt = full_rosbag_dataset[full_data_index][0]
        new_data_opt = full_rosbag_dataset[full_data_index][1]
        print("New data comes as input:", new_data_ipt, ", output:", new_data_opt)
        
        coreset.append(new_data_ipt, new_data_opt)
        full_data_index = full_data_index + 1

        torch.save(model, os.path.join(args.model_save_dir, "model.pt"))

def train(args, model, diter, dl, iteration, cycle, criteria, optimizer, writer, device):
    for iter_tmp in range(args.iteration):
        try:
            c_n, c_r = next(diter)
        except StopIteration:
            diter = iter(dl)
            c_n, c_r = next(diter)
        
        c_n = c_n.type(torch.FloatTensor)
        c_n = torch.squeeze(c_n).to(device)
        c_r = c_r.type(torch.FloatTensor)
        
        c_n = c_n.to(device)
        c_r = c_r.to(device)

        output = model(c_n).squeeze()

        loss = criteria(output, c_r)
        loss.retain_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar("loss", loss.item(), cycle*args.iteration+iter_tmp)

        if iter_tmp % (args.iteration/5) == 0:
            print("Loss:", loss.item())


if __name__ == '__main__':
    main()