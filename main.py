import numpy as np
import torch
from algo.ART import ART
from algo.DatasetBuffer import DatasetBuffer
from dataparser.dataparser import raw_data_parser, inference_data_parser, eval_data_parser
from arguments import get_args
from torch.utils.data import DataLoader
from torchsampler import ImbalancedDatasetSampler
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import os
from custom_utils import get_date, get_time
import matplotlib as mpl
import shutil
import torch.nn as nn
import torch.optim as optim
import csv
import wandb

is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')

now_date = get_date()
now_time = get_time()

def main():
    wandb.init()
    args = get_args()
    wandb.config.update(args)
    softmax = nn.Softmax(dim=1)

    wandb.run.name = str(now_date[2:]) + "_" + str(now_time)
    
    if os.path.exists('inferenced_result/'):
        shutil.rmtree('inferenced_result/')
        os.mkdir('inferenced_result/')

    model = ART(args.c_n_grid_size, args.c_n_grid_channel, args.c_r_class_num, args.dropout_rate, device)
    wandb.watch(model)

    if args.c_r_class_num == 1:
        loss_function = nn.MSELoss()
    elif args.c_r_class_num == 3:
        loss_function = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), args.lr)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.999999999999 ** epoch, last_epoch=-1, verbose=False)

    writer = SummaryWriter(log_dir='./runs/' + str(now_date[2:]) + "_" + str(now_time) + "_cr" + str(args.c_r_class_num) + "_cycle" + str(args.training_cycle) + "_coresize" + str(args.coreset_buffer_size))

    # Get full rosbag dataset for offline operation
    full_rosbag_dataset = raw_data_parser(args)
    eval_dataset_coreset = eval_data_parser(args)
    inference_dataset = inference_data_parser(args)

    # Coreset Initialization
    coreset = DatasetBuffer(args.coreset_buffer_size, args.coreset_type, args.c_r_class_num, "train")
    for i in range(args.coreset_buffer_size):
        coreset.append(full_rosbag_dataset[i][0], full_rosbag_dataset[i][1])
    
    full_data_index = args.coreset_buffer_size

    for cycle_idx in range(args.training_cycle):
        if args.c_r_class_num == 3:
            dl = DataLoader(coreset, batch_size=args.training_batch_size-1, shuffle=False, sampler=ImbalancedDatasetSampler(coreset), num_workers=0, pin_memory=False, drop_last=True)
        elif args.c_r_class_num == 1:
            dl = DataLoader(coreset, batch_size=args.training_batch_size-1, shuffle=True, sampler=None, num_workers=0, pin_memory=False, drop_last=True)
        diter = iter(dl)

        # Update coreset
        new_data_ipt = full_rosbag_dataset[full_data_index][0]
        new_data_opt = full_rosbag_dataset[full_data_index][1]
        new_data_ipt_tensor = torch.tensor(new_data_ipt)
        new_data_opt_tensor = torch.tensor(new_data_opt)
        
        model.train()

        with torch.no_grad():
            output_softmax_history = torch.empty((args.network_ensemble_cycle, 2, args.c_r_class_num))
            tmp_input = torch.cat((new_data_ipt_tensor, new_data_ipt_tensor), 0)
            tmp_input = tmp_input.type(torch.FloatTensor).to(device)

            for ensemble_cycle in range(args.network_ensemble_cycle):
                output = model(tmp_input).squeeze()
                output_softmax = softmax(output).detach()
                output_softmax_history[ensemble_cycle, :, :] = output_softmax

            output_std = torch.std(output_softmax_history, 0, True).squeeze()
            uncertainty = torch.sum(output_std).item() / (args.network_ensemble_cycle * args.training_batch_size * args.c_r_class_num)
            print("--------------------------------------------------------------")
            print("New data comes with uncertainty", uncertainty)

        # Train and update NN
        train(args, model, diter, dl, new_data_ipt_tensor, new_data_opt_tensor, args.iteration, cycle_idx, loss_function, optimizer, scheduler, writer, device)
        writer.flush()

        coreset.append(new_data_ipt, new_data_opt)
        print("Corseet updated with cycle", cycle_idx+1, "/", args.training_cycle, ". Mild:", len(coreset.c_n_mild), "Moderate:", len(coreset.c_n_moderate), "Harsh:", len(coreset.c_n_harsh))
        full_data_index = full_data_index + 1
        
        torch.save(model, os.path.join(args.model_save_dir, "model.pt"))
        evaluation(args, cycle_idx, model, eval_dataset_coreset, loss_function, writer, device)
        # inference(args, cycle_idx, model, inference_dataset)


def train(args, model, diter, dl, new_data_ipt_tensor, new_data_opt_tensor, iteration, cycle, criteria, optimizer, scheduler, writer, device):
    # model.train()
    for iter_tmp in range(args.iteration):
        try:
            c_n, c_r = next(diter)
        except StopIteration:
            diter = iter(dl)
            c_n, c_r = next(diter)

        c_n = torch.squeeze(c_n)
        
        if args.c_r_class_num == 3:
            if new_data_opt_tensor.size()[0] == 3:
                new_data_opt_tensor = torch.reshape(new_data_opt_tensor, (1, new_data_opt_tensor.size()[0]))

        elif args.c_r_class_num == 1:
            c_r = torch.reshape(c_r, (args.training_batch_size-1,1))
            new_data_opt_tensor = torch.reshape(new_data_opt_tensor, (1,1))
    
        c_n = torch.cat((c_n, new_data_ipt_tensor), 0)
        c_r = torch.cat((c_r, new_data_opt_tensor), 0)
        
        c_n = c_n.type(torch.FloatTensor)
        c_r = c_r.type(torch.FloatTensor)

        c_n = c_n.to(device)
        c_r = c_r.to(device)    
        
        softmax = nn.Softmax(dim=1)
        output_history = torch.empty((args.network_ensemble_cycle, args.training_batch_size, args.c_r_class_num))
        output_softmax_history = torch.empty((args.network_ensemble_cycle, args.training_batch_size, args.c_r_class_num))
        
        for ensemble_cycle in range(args.network_ensemble_cycle):
            output = model(c_n).squeeze()
            output_softmax = softmax(output).detach()
            output_history[ensemble_cycle, :, :] = output
            output_softmax_history[ensemble_cycle, :, :] = output_softmax

        output_mean = torch.mean(output_history, 0, True).squeeze().to(device)
        output_std = torch.std(output_softmax_history, 0, True).squeeze()
        uncertainty = torch.sum(output_std).item() / (args.network_ensemble_cycle * args.training_batch_size * args.c_r_class_num)
        
        # Compute Loss term
        loss = criteria(output_mean, c_r)

        # Compute Regularization Term
        if args.regularization_type == 'ucl':
            regularization_weight = np.exp(-1 * uncertainty) * args.ucl_weight
        elif args.regularization_type == 'vanilla':
            regularization_weight = 0.01
        else:
            regularization_weight = 0

        reg = torch.tensor(0.).to(device)

        for param in model.parameters():
            reg = reg + torch.norm(param)
        
        loss = loss + regularization_weight * reg

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        scheduler.step()
        wandb.log({
            "train_loss": loss.item(),
            "Uncertainty": uncertainty
        })
        writer.add_scalar("train_loss", loss.item(), cycle*args.iteration+iter_tmp)
        writer.add_scalar("Uncertainty", uncertainty)
        
        if iter_tmp % (args.iteration/5) == 0:
            print(str(now_date[2:]) + "_" + str(now_time), ") Training loss:", loss.item(), "Uncertainty:", uncertainty, "LR:", optimizer.param_groups[0]['lr'])


def evaluation(args, cycle, model, evaluation_dataset, criteria, writer, device):
    EVAL_BATCH_SIZE = 330
    dl = DataLoader(evaluation_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False, sampler=None, num_workers=0, pin_memory=False, drop_last=False)
    diter = iter(dl)
    right = 0
    wrong = 0
    model.eval()
    criteria.eval()
    with torch.no_grad():
        try:
            c_n, c_r = next(diter)
        except StopIteration:
            diter = iter(dl)
            c_n, c_r = next(diter)

        c_n = torch.squeeze(c_n)
        c_n = c_n.type(torch.FloatTensor)
        c_r = c_r.type(torch.FloatTensor)
        c_n = c_n.to(device)
        c_r = c_r.to(device)

        output = model(c_n).squeeze()
        loss = criteria(output, c_r)
        if args.c_r_class_num == 3:
            output_label = torch.argmax(output, dim=1)
            gt_c_r_label = torch.argmax(c_r, dim=1)
            

            for output_size in range(output_label.size()[0]):
                if output_label[output_size] == gt_c_r_label[output_size]:
                    right = right + 1
                else:
                    wrong = wrong + 1

            wandb.log({
                "eval_loss": loss.item(),
                "eval_accuracy": right / (right+wrong)
            })
            writer.add_scalar("eval_loss", loss.item(), cycle)
            writer.add_scalar("eval_accuracy", right / (right+wrong), cycle)
            print("EVAL ACCURACY", right / (right+wrong), "EVAL LOSS", loss.item())
        
        elif args.c_r_class_num == 1:
            writer.add_scalar("eval_loss", loss.item(), cycle)
            print("EVAL LOSS", loss.item())
    
def inference(args, cycle_idx, model, inference_dataset):
    INFERENCE_BATCH_SIZE = 2
    WINDOW_SIZE = 10
    LENGTH = 400
    C_R_GRID = np.zeros((INFERENCE_BATCH_SIZE, int(LENGTH/WINDOW_SIZE), int(LENGTH/WINDOW_SIZE)))

    dl = DataLoader(inference_dataset, batch_size=INFERENCE_BATCH_SIZE, shuffle=False, sampler=None, num_workers=0, pin_memory=False, drop_last=False)
    diter = iter(dl)

    model.eval()
    with torch.no_grad():
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



if __name__ == '__main__':
    main()