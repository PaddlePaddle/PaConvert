
import torch, six

import torch.nn

import torch.nn as nn, six as ss

from torch.nn import functional as F, init as I

from torch.nn import Module, Linear

from torch import add, Tensor

from io import open

from . import functional_pil as F_pil, functional_tensor as F_t

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
    def func5(x):
        return torch.transpose(x, 1, 0)

    return torch.abs(x)

if x > 1:
    y = x.transpose(0, 1)
else:
    z = x.transpose(0, 1)


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

# call api in different position
torch.abs(x, out = y)

return torch.abs(x, out = y)

z = torch.abs(x, out = y)

# multi level call api
torch.reshape(torch.add(torch.abs(x), y), [3])

torch.reshape(torch.add(x.abs(), y), [3])

torch.reshape(torch.abs(x).add(y), [3])

torch.add(torch.abs(x), y).reshape(3)

torch.abs(x).add(y).reshape(3)

torch.add(x.abs(), y).reshape(3)

torch.reshape(x.abs().add(y), [3])

x.abs().add(y).reshape([3])

# multi level call
nn.CrossEntropyLoss().cuda(args.gpu)

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

blocks = []
blocks.append(('block1', torch.nn.Linear(10, 10)))
blocks.append(('block2', torch.nn.Linear(10, 10)))
nn.Sequential(OrderedDict(blocks))

blocks = []
blocks.append(torch.nn.Linear(10, 10))
blocks.append(torch.nn.Linear(10, 10))
nn.Sequential(*blocks)


nn.Sequential(nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=2, padding=1, groups=in_dim),)

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

# torch.Tensor.size()
size = x.size()

size = torch.abs(x, out=y).size()

x.abs().size()

## NonTorchClass
x.size[2]
## TorchClass: torch.Tensor
x.shape[2]

# torch.Tensor.Attribute
shape = x.shape

device = x.device

dtype = x.dtype

y = torch.abs(x).T

shape = torch.abs(x).shape

torch.abs(x).shape

# different kinds of torch.Tensor method 
z = (torch.triu(torch.ones(sz, sz)) == 1).abs()

(x + y).abs()

(x == y).abs()

(-x).abs()


# torch.Tensor.reshape(*shape)

x.reshape(2, 3)

x.reshape([2, 2])

x.reshape(shape=[2, 3])


# torch.max/min
torch.max(image)

torch.max(image, dim=1)

torch.max(input=image, other=label)

torch.min(image)

torch.min(image, dim=1)

torch.min(image, label)

torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)

# torch.rand
m = 2
n = 3

torch.rand(m, n)

torch.randn(2+3, 3, out = y)

torch.zeros(m+n, n, out = y, dtype=torch.float32, requires_grad=True)

torch.ones(2, 3, requires_grad=False)

torch.empty(m, n, pin_memory=True)

torch.full(2, 3, device=torch.device('cpu'), pin_memory=False)

torch.rand(2, 3)

torch.rand((2, 3))

torch.rand([2, 3])

torch.rand(size=(2, 3))

torch.rand(size=[2, 3])

torch.rand(())


torch.rand([])

# torch.randint

torch.randint(10, [2, 2])

torch.randint(2, 10, [2, 2])


# torch.Tensor.size
torch.abs(x).size()

torch.abs(x).size(2)

x.size()

x.size(0)


# torch.Tensor.item
torch.abs(x).item()

x.item()

# requires_grad / requires_grad_
assert not label.requires_grad

label_requires_grad = label.requires_grad_(True)
label = label.requires_grad_(False)

requires_grad = [True, False]

label.requires_grad_(requires_grad[1])

assert label.requires_grad
assert label_requires_grad.requires_grad

# torch.einsum
torch.einsum("...ijk, ...xijk -> ...xjk", mask, a4)

# if scope insert
if pic.mode == '1':
    img = 255 * img

if pic.mode == '1':
    img = 255 * img


return torch.from_numpy(nppic).to(dtype=default_float_dtype)

# torch.Tensor.permute
x.permute(2, 3)

x.permute([2, 3])

x.permute(dims=[2, 3])


# torch.Tensor.repeat
import numpy as np
np.array([2., 3.]).repeat(2, axis = 0)

x.repeat(2, 3)

x.repeat([2, 3])

x.repeat(repeats=[2, 3])

# torch.Tensor.view
import torch
import numpy
import numpy as np

x = torch.rand([2, 3])

x.view(np.int32)

x.view(numpy.int32)

x.view(3, 2)

x.view([3, 2])

x.view(torch.int32)

# torch.Tensor.to
x.to(torch.float64)

cuda0 = torch.device('cuda:0')
x.to(cuda0)

other = torch.randn((), dtype=torch.float64, device=cuda0)
x.to(other, non_blocking=True)

# torch.Tensor.int/long/float/double
x=torch.rand(2, 3)
x.float()
x.double()
x.int()
x.long()


# torch.Tensor.type_as/type
x=torch.rand(2, 3)
y=torch.rand(2, 3)

x.type(torch.float32)
x.type(torch.float32, non_blocking=True)

x.type_as(y)
x.type_as(tensor=y)


# torch.nn.functional.interpolate
torch.nn.functional.interpolate(input_data, scale_factor=[2,1])

torch.nn.functional.interpolate(input_data, scale_factor=[2,1], recompute_scale_factor=True)

torch.nn.functional.interpolate(input_data, scale_factor=[2,1], recompute_scale_factor=False)

torch.nn.functional.interpolate(input_data, scale_factor=[2,1], antialias=False)


# torch.tensor
device = torch.device('cuda')
torch.tensor(1., device=device)

torch.tensor(1., device=torch.device('cuda:1'))

torch.tensor(1., device='cuda')

torch.tensor(1., device='cuda:1')

torch.tensor(1., requires_grad=True)

# torch.as_tensor
##### TODO: device cuda:1 can not support
torch.as_tensor(1., dtype=torch.float32, device=torch.device('cuda:0'))

# should not convert
import numpy as np

from np import array

np.add(x, y)
array(1.).abs().add(y)

# should mark unspport
torch.abs(x)
# should not mark unspport
( array(1.) + array(2.)).abs()
( array(1.) - array(2.)).abs()
( array(1.) * array(2.).numpy()).abs()
"_torch.npy"
str1="_torch.npy"
str2='_torch.npy'
hellotorch.test

## should mark
torch.save('torch.parma')
## not mark
np.save('torch.parma')

# torch.tensor/paddle.to_tensor
torch.tensor(features_A).T.cuda()


# torch.Tensor.transpose

## not torch Tensor
all_dists = dists.transpose()

## is torch Tensor
all_dists = dists.transpose(0, 1)


# Module class method
import torch.nn as nn

nn.CrossEntropyLoss().to(torch.device('cuda'))


linear = torch.nn.Linear(10, 10)

state_dict = linear.state_dict()

linear.load_state_dict(state_dict)

linear.parameters()

linear.named_parameters()

linear.buffers()

linear.named_buffers()

linear.children()

linear.named_children()

linear.modules()

linear.named_modules()

linear.train()

linear.eval()

linear.requires_grad_()

linear.zero_grad()

# Optimizer class method

sgd = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

state_dict = sgd.state_dict()

sgd.load_state_dict(state_dict)

sgd.zero_grad()

sgd.step()


