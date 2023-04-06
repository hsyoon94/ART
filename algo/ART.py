import torch
import torch.nn as nn

class ART(nn.Module):
    def __init__(self, input_channel_num, output_dim, dropout_rate, training_batch_size, experiment, device):
        super(ART, self).__init__()
        self.input_channel_num = input_channel_num
        self.output_dim = output_dim
        self.training_batch_size = training_batch_size
        self.experiment = experiment
        self.device = device
        self.cnn_mid_channel = 128
        self.cnn_final_channel = training_batch_size
        
        self.ReLU = nn.ReLU().to(self.device)
        self.Softmax = nn.Softmax(dim=0)
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(self.dropout_rate)

        self.CNN1 = nn.Conv2d(self.input_channel_num, self.cnn_mid_channel, 3, stride=1).to(self.device)
        self.CNN2 = nn.Conv2d(self.cnn_mid_channel, self.cnn_final_channel, 3, stride=1).to(self.device)
        self.MaxPool = nn.MaxPool2d(3, 1).to(self.device)
        self.batchnorm = nn.BatchNorm2d(self.training_batch_size).to(self.device)

        if self.experiment == 'husky':
            self.f1_input_dim = 128 * int(self.training_batch_size / 8)
        elif self.experiment == 'mnist':
            self.f1_input_dim = 3872
        elif self.experiment == 'cifar10':
            self.f1_input_dim = 5408
        elif self.experiment == 'cifar100':
            self.f1_input_dim = 5408
        elif self.experiment == 'imagenet':
            self.f1_input_dim = 5408

        # For regression task NN
        if self.output_dim == 1:
            self.F1 = nn.Linear(self.f1_input_dim, self.output_dim).to(self.device)
        else:
            self.F1 = nn.Linear(self.f1_input_dim, 64).to(self.device)
            self.F2 = nn.Linear(64, 32).to(self.device)
            self.F3 = nn.Linear(32, self.output_dim).to(self.device)
        
    def forward(self, input):
        output = 0
        if self.output_dim == 1:
            output = self.CNN(input)
            output = self.MaxPool(output)
            output = torch.flatten(output, start_dim=1)
            output = self.F1(output)
        elif self.output_dim != 1:
            output = self.CNN1(input)
            output = self.ReLU(output)
            output = self.CNN2(output)
            output = self.batchnorm(output)
            output = self.ReLU(output)
            output = self.MaxPool(output)
            output = torch.flatten(output, start_dim=1)
            output = self.F1(output)
            output = self.ReLU(output)
            output = self.dropout(output)
            output = self.F2(output)
            output = self.ReLU(output)
            output = self.F3(output)
            output = self.ReLU(output)

        return output