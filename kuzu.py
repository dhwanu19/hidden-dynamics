"""
   kuzu.py
   COMP9444, CSE, UNSW
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class NetLin(nn.Module):
    # linear function followed by log_softmax
    def __init__(self):
        super(NetLin, self).__init__()
        # 28*28 features (pixels)
        # 10 classes (outputs)
        self.in_to_out = torch.nn.Linear(28*28, 10)

    def forward(self, x):
        # data dimensions = n_obs * (width * length)
        # we are using a 1D linear unit that takes n_obs * n_features
        x_flattened = torch.flatten(input=x, start_dim=1) # dim=1
        net = self.in_to_out(x_flattened)
        # dim = A dimension along which LogSoftmax will be computed.
        # data dimensions = n_features * n_classes
        # set dim=1 since we want the prob per class
        output = F.log_softmax(net, dim=1)
        return output

class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self):
        super(NetFull, self).__init__()
        n_nodes = 150
        self.in_to_h1 = torch.nn.Linear(28*28, n_nodes)
        self.h1_to_out = torch.nn.Linear(n_nodes, 10)

    def forward(self, x):      
        x_flattened = torch.flatten(input=x, start_dim=1) # dim=1
        net_h1 = self.in_to_h1(x_flattened)
        
        out_h1 = F.tanh(net_h1)
        net_out = self.h1_to_out(out_h1)
        
        output = F.log_softmax(net_out, dim=1)
        return output
        

class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        # inputDim = 28
        # outputDim = (28 - 3)/1 + 1 = 26
        self.in_to_c1 = torch.nn.Conv2d(in_channels=1, out_channels=20, kernel_size=3, stride=1)
        
        # inputDim = 26
        # outputDim = (26 - 4) / 2 + 1 = 12
        self.pool_c1 = nn.MaxPool2d(kernel_size=4, stride=2)
        
        # inputDim = 12
        # outputDim = (12 - 3) / 1 + 1 = 10
        self.c1_to_c2 = torch.nn.Conv2d(in_channels=20, out_channels=40, kernel_size=3, stride=1)
        
        # inputDim = 10
        # outputDim = (10 - 4) / 2 + 1 = 4
        self.pool_c2 = nn.MaxPool2d(kernel_size=4, stride=2)
                
        # inputDim = 40*4*4 (flattened)
        # outputDim set to 144
        self.c2_to_fc1 = torch.nn.Linear(in_features=40*4*4, out_features=144)
        
        # inputDim = 144
        # outputDim set to 10
        self.fc1_to_out = torch.nn.Linear(in_features=144, out_features=10)

    def forward(self, x):
        net_c1 = self.in_to_c1(x)
        out_c1 = F.relu(net_c1)
        out_c1_p = self.pool_c1(out_c1)
        
        net_c2 = self.c1_to_c2(out_c1_p)
        out_c2 = F.relu(net_c2)        
        out_c2_p = self.pool_c2(out_c2)

        out_c2_f = torch.flatten(out_c2_p, start_dim=1)
        net_fc1 = self.c2_to_fc1(out_c2_f)
        out_fc1 = F.relu(net_fc1)
        
        net_out = self.fc1_to_out(out_fc1)
        output = F.log_softmax(net_out, dim=1)
        return output
