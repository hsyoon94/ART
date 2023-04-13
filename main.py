import numpy as np
import torch
from algo.ART import ART
from algo.DatasetBuffer import ARTBuffer
from algo.training import train, train_offline, evaluation, inference
from dataparser.dataparser import raw_data_parser, inference_data_parser, eval_data_parser
from arguments import get_args
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchsampler import ImbalancedDatasetSampler
from PIL import Image
import os
from utils.custom_utils import get_date, get_time
import matplotlib as mpl
import shutil
import torch.nn as nn
import torch.optim as optim
import csv
import wandb
import random

is_cuda = torch.cuda.is_available()
if torch.cuda.device_count() > 1:
    device = torch.device('cuda:2' if is_cuda else 'cpu')
else:
    device = torch.device('cuda:0' if is_cuda else 'cpu')

now_date = get_date()
now_time = get_time()

def main():
    args = get_args()
    if args.experiment == 'mnist':
        args.c_n_grid_channel = 1
        args.c_r_class_num = 10
    elif args.experiment == 'cifar10':
        args.c_n_grid_channel = 3
        args.c_r_class_num = 10
    elif args.experiment == 'cifar100':
        args.c_n_grid_channel = 3
        args.c_r_class_num = 100

    experiment_info = str(now_date[2:]) + "_" + str(now_time) + "_" + str(args.model) + "_" + str(args.experiment) + "_reg:" +  str(args.regularization_type) + "(" + str(args.reg_lambda) + ")_ctype:" + str(args.coreset_type) + "(" + str(args.greedy_degree) + ")"

    wandb.init()
    wandb.config.update(args)
    wandb.run.name = str(now_date[2:]) + "_" + str(now_time)
    evaluation_accuracy_history = list()
    evaluation_accuracy_after_coreset_update_algo_operation_history = list()
    softmax = nn.Softmax(dim=1)

    if args.model == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(args.dropout_rate), nn.Linear(num_ftrs, args.c_r_class_num))
        model = model.to(device)
    elif args.model == 'art':
        model = ART(args.c_n_grid_channel, args.c_r_class_num, args.dropout_rate, args.training_batch_size, args.experiment, device)
    wandb.watch(model)

    if args.c_r_class_num == 1:
        loss_function = nn.MSELoss()
    else:
        loss_function = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), args.lr)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.9999 ** epoch, last_epoch=-1, verbose=False)

    if args.experiment == 'husky':
        # Get full husky dataset
        train_dataset = raw_data_parser(args)
        eval_dataset = eval_data_parser(args)
        # inference_dataset = inference_data_parser(args)

    elif args.experiment == "mnist":
        train_dataset = datasets.MNIST('dataparser/dataset/',train = True,download = True,transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5), (0.5))]))
        eval_dataset = datasets.MNIST('dataparser/dataset/',train = False,download = True,transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5), (0.5))]))
        args.coreset_buffer_size = 1000

    elif args.experiment == "cifar10":
        train_dataset = datasets.CIFAR10('dataparser/dataset/',train = True,download = True,transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        eval_dataset = datasets.CIFAR10('dataparser/dataset/',train = False,download = True,transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        args.coreset_buffer_size = 1000

    elif args.experiment == "cifar100":
        train_dataset = datasets.CIFAR100('dataparser/dataset/',train = True,download = True,transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        eval_dataset = datasets.CIFAR100('dataparser/dataset/',train = False,download = True,transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        args.coreset_buffer_size = 10000

    if args.offline_learning is True:
        for cycle_idx in range(args.training_cycle):
            model.train()
            train_offline(args, model, train_dataset, loss_function, optimizer, scheduler, device)
            eval_acc = evaluation(args, 0, model, eval_dataset, loss_function, device)
            evaluation_accuracy_history.append(eval_acc)

    elif args.offline_learning is False:
        coreset = ARTBuffer(args.coreset_buffer_size, args.coreset_type, args.c_r_class_num, args.experiment, "train")
        rnd_idx = random.sample(range(len(train_dataset)), len(train_dataset))
        CORESET_INITIALIZE_SIZE = int(args.coreset_buffer_size * 0.95)

        for i in range(CORESET_INITIALIZE_SIZE):
            coreset.initial_append(train_dataset[rnd_idx[i]][0], train_dataset[rnd_idx[i]][1], model, args.network_ensemble_cycle)
        
        full_data_index = CORESET_INITIALIZE_SIZE

        for cycle_idx in range(args.training_cycle):
            new_data_ipt = list()
            new_data_opt = list()
            # Update coreset
            for x in range(args.streaming_data_size):
                new_data_ipt.append(train_dataset[rnd_idx[full_data_index + x]][0])
                new_data_opt.append(train_dataset[rnd_idx[full_data_index + x]][1])

            new_data_ipt_tensor = torch.stack(new_data_ipt, dim=0)
            new_data_opt_tensor = torch.tensor(new_data_opt)

            # Train and update NN
            model.train()
            train(args, model, coreset, new_data_ipt_tensor, new_data_opt_tensor, args.iteration, cycle_idx, loss_function, optimizer, scheduler, device)
            
            if new_data_ipt_tensor is not None:
                coreset.append(new_data_ipt, new_data_opt, model, args.network_ensemble_cycle)
            print("===============================================================================================================")
            
            print("Exp info:", experiment_info)
            print("Corseet updated with cycle", cycle_idx+1, "/", args.training_cycle)
            print()
            for class_idx in range(args.c_r_class_num):
                print(" Class" + str(class_idx), end="    ")
            print()
            for class_idx in range(args.c_r_class_num):
                print("  ", len(coreset.c_n_classwise[class_idx]), end="      ")
            print("\n")

            full_data_index = full_data_index + args.streaming_data_size
            
            eval_acc = evaluation(args, cycle_idx, model, eval_dataset, loss_function, device)
            evaluation_accuracy_history.append(eval_acc)

            if coreset.coreset_management_start is True:
                evaluation_accuracy_after_coreset_update_algo_operation_history.append(eval_acc)
            print("===============================================================================================================\n\n")
            # inference(args, cycle_idx, model, inference_dataset)

    print("========== Training Ends  ===========")
    # Experiment logging...
    evaluation_accuracy_history_np = np.array(evaluation_accuracy_history)
    value_mean = np.mean(evaluation_accuracy_history_np)
    value_std = np.std(evaluation_accuracy_history_np)

    evaluation_accuracy_after_coreset_update_algo_operation_history_np = np.array(evaluation_accuracy_after_coreset_update_algo_operation_history)
    value_after_mean = np.mean(evaluation_accuracy_after_coreset_update_algo_operation_history_np)
    value_after_std = np.std(evaluation_accuracy_after_coreset_update_algo_operation_history_np)
    
    max_acc = np.max(evaluation_accuracy_history_np)
    max_acc_idx = np.argmax(evaluation_accuracy_history_np)
    forgetting_score_overall = 0.0
    forgetting_score_after = 0.0
    for history_idx in range(len(evaluation_accuracy_history)):
        forgetting_score_overall = forgetting_score_overall + (max_acc - evaluation_accuracy_history_np[history_idx])
    forgetting_score_overall = forgetting_score_overall / len(evaluation_accuracy_history)

    for history_idx2 in range(max_acc_idx, len(evaluation_accuracy_history)):
        forgetting_score_after = forgetting_score_after + (max_acc - evaluation_accuracy_history_np[history_idx])
    forgetting_score_after = forgetting_score_after / (len(evaluation_accuracy_history) - max_acc_idx)
    
    args.evaluation_accuracy_mean = value_mean
    args.evaluation_accuracy_std = value_std

    args.evaluation_accuracy_after_mean = value_after_mean
    args.evaluation_accuracy_after_std = value_after_std

    args.forgetting_score_overall = forgetting_score_overall
    args.forgetting_score_after = forgetting_score_after
    
    wandb.config.update(args, allow_val_change=True)

if __name__ == '__main__':
    main()