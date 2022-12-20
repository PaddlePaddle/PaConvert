
import torch, six

import torch.nn

import torch.nn as nn, six as s

from torch.nn import functional as F, init as I

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
    def forward(self, x):
        x = torch.rand(10, 10)

        return torch.transpose(x, 1, 0)

class MyNet2(torch.nn.Module):
    pass

@torch.no_grad()
def func1(x):
    return torch.abs(x)

def func2(x) -> torch.Tensor:
    return torch.abs(x)

def func3(x: torch.Tensor) -> torch.Tensor:
    def func5():
        return torch.transpose(x, 1, 0)

    return torch.abs(x)

if x > 1:
    print("true")
else:
    print("false")

def func4(x: Tensor=None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        torch.add(torch.rand(1, 2, requires_grad=True), torch.rand(1, 2, requires_grad=True))
        return torch.transpose(x, 1, 0)

linear = MyNet()

x = torch.rand(10, 10)

y = torch.transpose(x, 1, 0)

y = x.transpose(1, 0)

y_shape = x.transpose(1,0).shape

z = linear(y)

out = torch.rand(1, 2, 3, dtype=torch.float32)

torch.rand(1, 2, 3, dtype=torch.float32, requires_grad=True)

torch.tensor(1., requires_grad=True)

# call api in different position
torch.abs(x, out = y)

return torch.abs(x, out = y)

z = torch.abs(x, out = y)

# multi level call api
torch.reshape(torch.add(torch.abs(x), y), [3])

torch.reshape(torch.add(x.abs(), y), [3])

torch.reshape(torch.abs(x).add(y), [3])

torch.add(torch.abs(x), y).reshape([3])

torch.abs(x).add(y).reshape([3])

torch.add(x.abs(), y).reshape([3])

torch.reshape(x.abs().add(y), [3])

x.abs().add(y).reshape([3])

# Sequential
model = nn.Sequential(
          nn.Conv2d(1,20,5),
          nn.ReLU(),
          nn.Conv2d(20,64,5),
          nn.ReLU()
        )

model = nn.Sequential(OrderedDict([
          ('conv1', nn.Conv2d(1,20,5)),
          ('relu1', nn.ReLU()),
          ('conv2', nn.Conv2d(20,64,5)),
          ('relu2', nn.ReLU())
        ]))

# container
linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])

layers = nn.ModuleDict({
                    'conv': nn.Conv2d(10, 10, 3),
                    'pool': nn.Conv2d(10, 10, 3)
                })

# torch.nn.GELU
gelu = torch.nn.GELU(approximate='tanh')

torch.nn.GELU()

torch.nn.GELU(approximate='none')

F.gelu(x, approximate='none')

y = F.gelu(x, approximate='none')
