import numpy as np
import torch
from algo.ART import ART
from algo.DatasetBuffer import DatasetBuffer
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
device = torch.device('cuda' if is_cuda else 'cpu')

now_date = get_date()
now_time = get_time()

def main():
    args = get_args()
    wandb.init()
    wandb.config.update(args)
    wandb.run.name = str(now_date[2:]) + "_" + str(now_time)
    evaluation_accuracy_history = list()
    softmax = nn.Softmax(dim=1)

    if args.model == 'resnet18':
        model = models.resnet18(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, args.c_r_class_num)
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
        inference_dataset = inference_data_parser(args)

    elif args.experiment == "mnist":
        train_dataset = datasets.MNIST('dataparser/dataset/',train = True,download = True,transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5), (0.5))]))
        eval_dataset = datasets.MNIST('dataparser/dataset/',train = False,download = True,transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5), (0.5))]))

    elif args.experiment == "cifar10":
        train_dataset = datasets.CIFAR10('dataparser/dataset/',train = True,download = True,transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        eval_dataset = datasets.CIFAR10('dataparser/dataset/',train = False,download = True,transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

    elif args.experiment == "cifar100":
        train_dataset = datasets.CIFAR100('dataparser/dataset/',train = True,download = True,transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        eval_dataset = datasets.CIFAR100('dataparser/dataset/',train = False,download = True,transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

    if args.offline_learning is True:
        for cycle_idx in range(args.training_cycle):
            model.train()
            train_offline(args, model, train_dataset, loss_function, optimizer, scheduler, device)
            eval_acc = evaluation(args, 0, model, eval_dataset, loss_function, device)
            evaluation_accuracy_history.append(eval_acc)

    elif args.offline_learning is False:
        coreset = DatasetBuffer(args.coreset_buffer_size, args.coreset_type, args.c_r_class_num, args.experiment, "train")
        rnd_idx = random.sample(range(len(train_dataset)), len(train_dataset))
        for i in range(args.coreset_buffer_size):
            coreset.append(train_dataset[rnd_idx[i]][0], train_dataset[rnd_idx[i]][1], None, 0)
        
        full_data_index = args.coreset_buffer_size

        for cycle_idx in range(args.training_cycle):
            
            # Update coreset
            new_data_ipt = train_dataset[rnd_idx[full_data_index]][0]
            new_data_opt = train_dataset[rnd_idx[full_data_index]][1]

            if torch.is_tensor(new_data_ipt) is False:
                new_data_ipt_tensor = torch.tensor(new_data_ipt)
                new_data_opt_tensor = torch.tensor(new_data_opt)
            else:
                new_data_ipt_tensor = new_data_ipt.clone().detach()
                new_data_opt_tensor = torch.tensor(new_data_opt).clone().detach()

            # Train and update NN
            model.train()
            train(args, model, coreset, new_data_ipt_tensor, new_data_opt_tensor, args.iteration, cycle_idx, loss_function, optimizer, scheduler, device)
            
            if new_data_ipt_tensor is not None:
                coreset.append(new_data_ipt, new_data_opt, model, args.network_ensemble_cycle)
            print("===============================================================================================================")
            print("Exp time:", str(now_date[2:]) + "_" + str(now_time))
            print("Corseet updated with cycle", cycle_idx+1, "/", args.training_cycle, ". Class", int(new_data_opt_tensor.item()), "appended")
            print()
            for class_idx in range(args.c_r_class_num):
                print(" Class" + str(class_idx), end="    ")
            print()
            for class_idx in range(args.c_r_class_num):
                print("  ", len(coreset.c_n_classwise[class_idx]), end="      ")
            print("\n")

            full_data_index = full_data_index + 1
            
            torch.save(model, os.path.join(args.model_save_dir, "model.pt"))
            eval_acc = evaluation(args, cycle_idx, model, eval_dataset, loss_function, device)
            evaluation_accuracy_history.append(eval_acc)
            print("===============================================================================================================\n\n")
            # inference(args, cycle_idx, model, inference_dataset)

        
    print("========== Training Ends  ===========")
    evaluation_accuracy_history_np = np.array(evaluation_accuracy_history)
    value_mean = np.mean(evaluation_accuracy_history_np)
    value_std = np.std(evaluation_accuracy_history_np)
    print("Eval accuracy mean:", value_mean)
    print("Eval accuracy std:", value_std)
    eval_result_plot_dir = os.path.join('./inferenced_result/evaluation_acc/', str(now_date[2:]) + "_" + str(now_time)+".txt")
    with open(eval_result_plot_dir, 'w') as file:
        file.write(str(value_mean)+"\n")
        file.write(str(value_std))




if __name__ == '__main__':
    main()