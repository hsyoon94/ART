import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb

def train_offline(args, model, dataset, criteria, optimizer, scheduler, device):
    model.train()
    
    # dl = DataLoader(dataset, batch_size=args.training_batch_size, shuffle=False, sampler=ImbalancedDatasetSampler(dataset), num_workers=0, pin_memory=False, drop_last=True)
    dl = DataLoader(dataset, batch_size=args.training_batch_size, shuffle=True, sampler=None, num_workers=0, pin_memory=False, drop_last=True)
    diter = iter(dl)

    for iter_tmp in range(args.iteration):
        try:
            c_n, c_r = next(diter)
        except StopIteration:
            diter = iter(dl)
            c_n, c_r = next(diter)

        if args.experiment != 'mnist':
            c_n = torch.squeeze(c_n)
        # c_n = torch.squeeze(c_n)
        c_r = torch.squeeze(c_r)
        c_n = c_n.type(torch.FloatTensor)
        c_r = c_r.type(torch.LongTensor)

        c_n = c_n.to(device)
        c_r = c_r.to(device)
        
        output = model(c_n).squeeze()
        # Compute Loss term
        loss = criteria(output, c_r)

        # Compute Regularization Term
        if args.regularization_type == 'l2':
            regularization_weight = args.reg_lambda
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
        })


def train(args, model, coreset, new_data_ipt_tensor, new_data_opt_tensor, iteration, cycle, criteria, optimizer, scheduler, device):
    
    if args.c_r_class_num != 1:
        # dl = DataLoader(coreset, batch_size=args.training_batch_size-1, shuffle=False, sampler=ImbalancedDatasetSampler(coreset), num_workers=0, pin_memory=False, drop_last=True)
        dl = DataLoader(coreset, batch_size=args.training_batch_size-1, shuffle=True, sampler=None, num_workers=0, pin_memory=False, drop_last=True)
    elif args.c_r_class_num == 1:
        dl = DataLoader(coreset, batch_size=args.training_batch_size-1, shuffle=True, sampler=None, num_workers=0, pin_memory=False, drop_last=True)
    diter = iter(dl)

    if args.c_r_class_num != 1:
        if args.experiment == 'husky':
            if new_data_opt_tensor.size()[0] == 10:
                new_data_opt_tensor = torch.reshape(new_data_opt_tensor, (1, new_data_opt_tensor.size()[0]))
        elif args.experiment == 'cifar10':
            new_data_ipt_tensor = torch.reshape(new_data_ipt_tensor, (1, new_data_ipt_tensor.size()[0], new_data_ipt_tensor.size()[1], new_data_ipt_tensor.size()[2]))
            new_data_opt_tensor = torch.unsqueeze(new_data_opt_tensor, 0)

    elif args.c_r_class_num == 1:
        c_r = torch.reshape(c_r, (args.training_batch_size-1,1))
        new_data_opt_tensor = torch.reshape(new_data_opt_tensor, (1,1))

    for iter_tmp in range(args.iteration):
        try:
            c_n, c_r = next(diter)
        except StopIteration:
            diter = iter(dl)
            c_n, c_r = next(diter)

        c_n = torch.squeeze(c_n)
        c_r = torch.squeeze(c_r)
        
        c_n = torch.cat((c_n, new_data_ipt_tensor), 0)
        c_r = torch.cat((c_r, new_data_opt_tensor), 0)

        c_n = c_n.type(torch.FloatTensor)
        c_r = c_r.type(torch.LongTensor)

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
            tau = 1
            regularization_weight = np.exp(1 * uncertainty / tau) * args.reg_lambda
        elif args.regularization_type == 'l2':
            regularization_weight = args.reg_lambda
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


def evaluation(args, cycle, model, evaluation_dataset, criteria, device):
    if args.experiment == 'husky':
        dl = DataLoader(evaluation_dataset, batch_size=len(evaluation_dataset), shuffle=False, sampler=None, num_workers=0, pin_memory=False, drop_last=False)
    else: 
        dl = DataLoader(evaluation_dataset, batch_size=100, shuffle=False, sampler=None, num_workers=0, pin_memory=False, drop_last=False)
    
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

        if args.experiment != 'mnist':
            c_n = torch.squeeze(c_n)

        c_n = c_n.type(torch.FloatTensor)
        if args.experiment == 'husky':
            c_r = c_r.type(torch.LongTensor).squeeze()
        elif args.experiment == 'cifar10' or args.experiment == 'cifar100':
            c_r = c_r.type(torch.LongTensor)
        c_n = c_n.to(device)
        c_r = c_r.to(device)

        output = model(c_n).squeeze()
        loss = criteria(output, c_r)
        if args.c_r_class_num != 1:
            output_label = torch.argmax(output, dim=1)
            try:
                gt_c_r_label = torch.argmax(c_r, dim=1)
            except IndexError:
                gt_c_r_label = c_r
            
            
            for output_size in range(output_label.size()[0]):
                if output_label[output_size] == gt_c_r_label[output_size]:
                    right = right + 1
                else:
                    wrong = wrong + 1

            wandb.log({
                "eval_loss": loss.item(),
                "eval_accuracy": right / (right+wrong)
            })
            print("EVAL ACCURACY", right / (right+wrong), "EVAL LOSS", loss.item())

        elif args.c_r_class_num == 1:
            print("EVAL LOSS", loss.item())
    
    return right / (right+wrong)


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