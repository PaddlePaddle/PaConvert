'''
import torch, six

import torch.nn

import torch.nn as nn, six as s

from torch.nn import functional as F, init as i

from torch.nn import Module, Linear

from torch import add, Tensor

class MyNet(nn.Module):
    test = "str"

    def __init__(self):
        self._fc1 = torch.nn.Linear(10, 10)
        self._fc2 = nn.Linear(10, 10)
        self._fc3 = Linear(10, 10)

    @torch.no_grad()
    def forward(self, x):
        x = self._fc1(x)
        x = self._fc2(x)
        x = self._fc3(x)
        y = add(x, x)
        return F.relu(y)

class MyNet1(Module):
    pass

class MyNet2(torch.nn.Module):
    pass

@torch.no_grad()
def func1(x):
    return torch.abs(x)

def func2(x) -> torch.Tensor:
    return torch.abs(x)

def func3(x: torch.Tensor) -> torch.Tensor:
    def func5():
        return True

    return torch.abs(x)

if x > 1:
    print("true")
else:
    print("false")

def func4(x: Tensor=None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return torch.abs(x)


linear = MyNet()

x = torch.rand([10, 10])

y = x.transpose(1, 0)

y_shape = x.transpose(1,0).shape

z = linear(y)


y = x.transpose(0, 2).reshape([2, 3])

torch.reshape(torch.transpose(x, 0, 2), [2, 3])


torch.reshape(x, [2, 3])


return x.transpose(0, 2).reshape([2, 3])
'''
