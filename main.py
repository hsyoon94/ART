import numpy as np
import torch
from algo.ART import ART
from algo.DatasetBuffer import DatasetBuffer
from dataparser.dataparser import raw_data_parser, inference_data_parser
from arguments import get_args
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import os

from custom_utils import get_date, get_time
import matplotlib as mpl
import shutil
import torch.nn as nn
import torch.optim as optim

import wandb
import csv

is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')
# device = 'cpu'

args = get_args()

def main():
    wandb.init(
    # set the wandb project where this run will be logged
    project="ART",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": args.lr,
    "c_r class num": args.c_r_class_num,
    "training cycle": args.training_cycle,
    "coreset size": args.coreset_buffer_size
    }
)

    if os.path.exists('inferenced_result/'):
        shutil.rmtree('inferenced_result/')
        os.mkdir('inferenced_result/')

    model = ART(args.c_n_grid_size, args.c_n_grid_channel, args.c_r_class_num, device)
    if args.c_r_class_num ==1 :
        loss_function = nn.MSELoss()
    else:
        loss_function = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), args.lr)
    now_date = get_date()
    now_time = get_time()
    writer = SummaryWriter(log_dir='./runs/' + str(now_date[2:]) + "_" + str(now_time) + "_cr" + str(args.c_r_class_num) + "_cycle" + str(args.training_cycle) + "_coresize" + str(args.coreset_buffer_size))

    # Get full rosbag dataset for offline operation
    full_rosbag_dataset = raw_data_parser(args)
    inference_dataset = inference_data_parser(args)

    # Coreset Initialization
    coreset = DatasetBuffer(args.coreset_buffer_size, args.coreset_type, args.c_r_class_num, "train")
    for i in range(args.coreset_buffer_size):
        coreset.append(full_rosbag_dataset[i][0], full_rosbag_dataset[i][1])

    full_data_index = args.coreset_buffer_size

    for cycle_idx in range(args.training_cycle):
        print("Training cycle", cycle_idx, "/", args.training_cycle, "starts")
        
        dl = DataLoader(coreset, batch_size=args.training_batch_size, shuffle=True, sampler=None, num_workers=0, pin_memory=False, drop_last=True)
        diter = iter(dl)

        # Train and update NN
        train(args, model, diter, dl, args.iteration, cycle_idx, loss_function, optimizer, writer, wandb, device)
        writer.flush()

        # Update coreset
        new_data_ipt = full_rosbag_dataset[full_data_index][0]
        new_data_opt = full_rosbag_dataset[full_data_index][1]
        print("New data comes as input:", new_data_ipt, ", output:", new_data_opt)
        
        coreset.append(new_data_ipt, new_data_opt)
        full_data_index = full_data_index + 1

        torch.save(model, os.path.join(args.model_save_dir, "model.pt"))
        inference(args, cycle_idx, model, inference_dataset)

def inference(args, cycle_idx, model, inference_dataset):
    INFERENCE_BATCH_SIZE = 2
    WINDOW_SIZE = 10
    LENGTH = 400
    C_R_GRID = np.zeros((INFERENCE_BATCH_SIZE, int(LENGTH/WINDOW_SIZE), int(LENGTH/WINDOW_SIZE)))

    dl = DataLoader(inference_dataset, batch_size=INFERENCE_BATCH_SIZE, shuffle=False, sampler=None, num_workers=0, pin_memory=False, drop_last=False)
    diter = iter(dl)

    for iter_tmp in range(INFERENCE_BATCH_SIZE):
        try:
            c_n = next(diter)
        except StopIteration:
            diter = iter(dl)
            c_n = next(diter)
        
        c_n = c_n.type(torch.FloatTensor)
        c_n = torch.squeeze(c_n).to(device)
        c_n = c_n.to(device)

        for row in range(int(LENGTH/WINDOW_SIZE)):
            for col in range(int(LENGTH/WINDOW_SIZE)):
                c_n_window = c_n[:, :, row*WINDOW_SIZE:(row+1)*WINDOW_SIZE, col*WINDOW_SIZE:(col+1)*WINDOW_SIZE]
                c_r_window = model(c_n_window).squeeze()
                c_r_max_index = torch.argmax(c_r_window, dim=1)
                
                for image_num in range(INFERENCE_BATCH_SIZE):
                    if c_r_max_index[image_num] == 0:
                        C_R_GRID[image_num, row, col] = 10
                    elif c_r_max_index[image_num] == 1:
                        C_R_GRID[image_num, row, col] = 50
                    elif c_r_max_index[image_num] == 2:
                        C_R_GRID[image_num, row, col] = 90

    C_R_GRID = C_R_GRID.astype(np.uint8)
    for image_num in range(INFERENCE_BATCH_SIZE):
        im = Image.fromarray(C_R_GRID[image_num,:,:])
        im = im.resize((400, 400), Image.BICUBIC)
        im.save("inferenced_result/image" + str(image_num) + "_" + str(cycle_idx).zfill(4) + "_.jpeg") 

def train(args, model, diter, dl, iteration, cycle, criteria, optimizer, writer, wandb, device):
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
        wandb.log({"loss": loss})
        
        if iter_tmp % (args.iteration/5) == 0:
            print("Loss:", loss.item())


if __name__ == '__main__':
    main()