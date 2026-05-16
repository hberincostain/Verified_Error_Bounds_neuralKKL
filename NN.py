"""
This file is from the MIT-licensed repository:

    Learning-based Design of Luenberger Observers for Nonlinear Systems
    Umar Niazi, John Cao, Xudong Sun, Amritam Das, and Karl Johansson
    Version: 1.0.0
    Released: 2022-10-04
    Repository: https://github.com/Mudhdhoo/ACC_KKL_Observer
"""

import torch
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MLP(nn.Module):
    def __init__(self, hidden_sizes, in_size, out_size, activation, input_std=torch.zeros(1), input_mean=torch.zeros(1), output_std=torch.zeros(1), output_mean=torch.zeros(1)):
        super().__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        self.input_std = input_std.float()
        self.input_mean = input_mean.float()
        self.output_std = output_std.float()
        self.output_mean = output_mean.float()
        self.mode = 'normal'
        current_dim = in_size
        for h_size in hidden_sizes:
            self.layers.append(nn.Linear(current_dim, h_size).to(device))
            current_dim = h_size
        self.layers.append(nn.Linear(current_dim, out_size).to(device))

    def forward(self, tensor):
        """
        Forward method of the NN.
        Normalizer will normalize input and denormalize the output
        """
        # normalize input
        tensor = (tensor - self.input_mean.float()) / self.input_std.float()
        for layer in self.layers[:-1]:
            tensor = self.activation(layer(tensor))
        tensor = self.layers[-1](tensor) # no activation on the last layer
        tensor = tensor*self.output_std + self.output_mean
        return tensor

class NN(nn.Module):
    def __init__(self, hidden_sizes, in_size, out_size, activation, normalizer=None):
        super().__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        self.normalizer = normalizer
        self.mode = 'normal'
        current_dim = in_size
        for h_size in hidden_sizes:
            self.layers.append(nn.Linear(current_dim, h_size).to(device))
            current_dim = h_size
        self.layers.append(nn.Linear(current_dim, out_size).to(device))

    def forward(self, tensor):
        """
        Forward method of the NN.
        Normalizer will normalize input and denormalize the output
        """
        # normalize input
        if self.normalizer != None:
            tensor = self.normalizer.Normalize(tensor, self.mode)
        for layer in self.layers[:-1]:
            tensor = self.activation(layer(tensor))
        tensor = self.layers[-1](tensor) # no activation on the last layer
        
        # denormalize the output
        if self.normalizer != None:
            tensor = self.normalizer.Denormalize(tensor, self.mode)

        return tensor

# NN for sequential training, not combined training
class Main_Network(nn.Module):
    def __init__(self, x_size, z_size, hidden_sizes, activation, normalizer=None):
        super().__init__()
        self.normalizer = normalizer
        self.net = NN(hidden_sizes, x_size, z_size, activation, normalizer)
        self.mode = 'normal'

    def forward(self, x):
        self.net.mode = self.mode
        output = self.net(x.to(device)) # output from NN

        if self.normalizer != None:
            norm_output = self.normalizer.Normalize(output, self.mode).float()
        else:
            norm_output = output
        
        return output, norm_output