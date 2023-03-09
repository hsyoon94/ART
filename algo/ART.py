import torch
import torch.nn as nn

class ART(nn.Module):
    def __init__(self, input_dim, input_channel_num, output_dim, device):
        super(ART, self).__init__()
        self.input_dim = input_dim
        self.input_channel_num = input_channel_num
        self.output_dim = output_dim
        
        self.device = device
        self.cnn_mid_channel = 8
        
        self.dropout= nn.Dropout(p=0.1)
        self.ReLU = nn.ReLU().to(self.device)
        self.Softmax = nn.Softmax(dim=0)

        self.CNN = nn.Conv2d(self.input_channel_num, self.cnn_mid_channel, 3, stride=1).to(self.device)
        self.MaxPool = nn.MaxPool2d(3, 1).to(self.device)
        

        # For regression task NN
        if self.output_dim == 1:
            self.F1 = nn.Linear(288, self.output_dim).to(self.device)
        else:
            self.F1 = nn.Linear(288, 32).to(self.device)
            self.F2 = nn.Linear(32, self.output_dim).to(self.device)
        
    def forward(self, input):
        output = 0

        if self.output_dim == 1:
            output = self.CNN(input)
            output = self.MaxPool(output)
            output = torch.flatten(output, start_dim=1)
            output = self.F1(output)
        else:
            output = self.CNN(input)
            output = self.MaxPool(output)
            output = torch.flatten(output, start_dim=1)
            output = self.F1(output)
            output = self.F2(output)
            # output = self.Softmax(output)
        
        return output