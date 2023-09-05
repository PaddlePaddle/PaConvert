import torch
from torch import nn
from torch.nn import Module

def add_module(self, module):
    self.add_module(str(len(self) + 1), module)
    
torch.nn.Module.add_module = add_module
nn.Module.add_module = add_module
Module.add_module = add_module

network = torch.nn.Sequential(
          torch.nn.Conv2d(1,20,5),
          torch.nn.Conv2d(20,64,5)
        )

network.add_module(torch.nn.ReLU())

print(network)
