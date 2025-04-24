# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
   isort:skip_file
"""
import torch
import torch.nn
import torch.nn as nn
from torch import Tensor, add
from torch.nn import Linear, Module
from torch.nn import functional as F


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


def func4(x: Tensor = None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        torch.add(
            torch.rand(1, 2, requires_grad=True), torch.rand(1, 2, requires_grad=True)
        )
        return torch.transpose(x, 1, 0)


linear = MyNet()

x = torch.rand(10, 10)

y = torch.transpose(x, 1, 0)

y = x.transpose(1, 0)

y_shape = x.transpose(1, 0).shape

z = linear(y)

out = torch.rand(1, 2, 3, dtype=torch.float32)

torch.rand(1, 2, 3, dtype=torch.float32, requires_grad=True)

# call api in different position
torch.abs(x, out=y)

return torch.abs(x, out=y)

z = torch.abs(x, out=y)

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
model = nn.Sequential(nn.Conv2d(1, 20, 5), nn.ReLU(), nn.Conv2d(20, 64, 5), nn.ReLU())

model = nn.Sequential(
    OrderedDict(
        [
            ("conv1", nn.Conv2d(1, 20, 5)),
            ("relu1", nn.ReLU()),
            ("conv2", nn.Conv2d(20, 64, 5)),
            ("relu2", nn.ReLU()),
        ]
    )
)

blocks = []
blocks.append(("block1", torch.nn.Linear(10, 10)))
blocks.append(("block2", torch.nn.Linear(10, 10)))
nn.Sequential(OrderedDict(blocks))

blocks = []
blocks.append(torch.nn.Linear(10, 10))
blocks.append(torch.nn.Linear(10, 10))
nn.Sequential(*blocks)


nn.Sequential(
    nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=2, padding=1, groups=in_dim),
)

# container
linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])

layers = nn.ModuleDict({"conv": nn.Conv2d(10, 10, 3), "pool": nn.Conv2d(10, 10, 3)})

# torch.nn.GELU
gelu = torch.nn.GELU(approximate="tanh")

torch.nn.GELU()

torch.nn.GELU(approximate="none")

F.gelu(x, approximate="none")

y = F.gelu(x, approximate="none")

# torch.Tensor.size()
size = x.size()

size = torch.abs(x, out=y).size()

x.abs().size()

image.size

image.size[2]

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

torch.randn(2 + 3, 3, out=y)

torch.zeros(m + n, n, out=y, dtype=torch.float32, requires_grad=True)

torch.ones(2, 3, requires_grad=False)

torch.empty(m, n, pin_memory=True)

torch.full([2, 3], 1.0, device=torch.device("cpu"), requires_grad=False)

torch.rand(2, 3)

torch.rand((2, 3))

torch.rand([2, 3])

torch.rand(size=(2, 3))

torch.rand(size=[2, 3])

torch.rand(())

torch.rand([])

torch.rand(2)

torch.rand(2, 3)

torch.rand([2, 3])

torch.rand((2, 3))

shape = 2
torch.rand(shape)

shape = (2, 3)
torch.rand(shape)

shape = [2, 3]
torch.rand(shape)

shape = (2, 3)
torch.rand(*shape)

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
if pic.mode == "1":
    img = 255 * img

if pic.mode == "1":
    img = 255 * img


return torch.from_numpy(nppic).to(dtype=default_float_dtype)

# torch.Tensor.permute
x.permute(2, 3)

x.permute([2, 3])

x.permute(dims=[2, 3])


# torch.Tensor.repeat
import numpy as np

np.array([2.0, 3.0]).repeat(2, axis=0)

x.repeat(2, 3)

x.repeat([2, 3])

x.repeat(repeats=[2, 3])

import numpy
import numpy as np

# torch.Tensor.view
import torch

x = torch.rand([2, 3])

x.view(np.int32)

x.view(numpy.int32)

x.view(3, 2)

x.view([3, 2])

x.view(torch.int32)

# torch.Tensor.to
x.to(torch.float64)

cuda0 = torch.device("cuda:0")
x.to(cuda0)

other = torch.randn((), dtype=torch.float64, device=cuda0)
x.to(other, non_blocking=True)

# torch.Tensor.int/long/float/double
x = torch.rand(2, 3)
x.float()
x.double()
x.int()
x.long()


# torch.Tensor.type_as/type
x = torch.rand(2, 3)
y = torch.rand(2, 3)

x.type(torch.float32)
x.type(torch.float32, non_blocking=True)

x.type_as(y)
x.type_as(tensor=y)


# torch.nn.functional.interpolate
torch.nn.functional.interpolate(input_data, scale_factor=[2, 1])

torch.nn.functional.interpolate(
    input_data, scale_factor=[2, 1], recompute_scale_factor=True
)

torch.nn.functional.interpolate(
    input_data, scale_factor=[2, 1], recompute_scale_factor=False
)

torch.nn.functional.interpolate(input_data, scale_factor=[2, 1], antialias=False)


# torch.tensor
device = torch.device("cuda")
torch.tensor(1.0, device=device)

torch.tensor(1.0, device=torch.device("cuda:1"))

torch.tensor(1.0, device="cuda")

torch.tensor(1.0, device="cuda:1")

torch.tensor(1.0)

torch.tensor(1.0, requires_grad=True)


# torch.as_tensor

torch.as_tensor(1.0, dtype=torch.float32, device=torch.device("cuda:0"))

# should not convert
import numpy as np
from np import array

np.add(x, y)
array(1.0).abs().add(y)

# should mark unsupport
torch.abs(x)
# should not mark unsupport
(array(1.0) + array(2.0)).abs()
(array(1.0) - array(2.0)).abs()
(array(1.0) * array(2.0).numpy()).abs()
"_torch.npy"
str1 = "_torch.npy"
str2 = "_torch.npy"
hellotorch.test

## should mark
torch.save(obj, "torch.parma")
## not mark
np.save("torch.parma")

# torch.tensor/paddle.to_tensor
torch.tensor(features_A).T.cuda()


# torch.Tensor.transpose

## not torch Tensor
all_dists = dists.transpose()

## is torch Tensor
all_dists = dists.transpose(0, 1)


# Module class method
import torch.nn as nn

nn.CrossEntropyLoss().to(torch.device("cuda"))

missing_keys, unexpected_keys = model_without_ddp.load_state_dict(
    checkpoint["model"], strict=False
)

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


# torch.Tensor.device
x.device

args.device

self.args.device

# some bad case
x = x + self.pos_embed.expand(B, -1, -1).detach()

(attn @ v).transpose(-2, -1)

x = self.proj(x).flatten(2).transpose(1, 2)

lt = torch.max(boxes1[:, None, :2])

lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])

lt = torch.min(boxes1[:, None, :2])

lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])

torch.sum(input, dim=1)

torch.sum(input, 1, dtype=torch.float32)

torch.mean(input, 1, dtype=torch.float32)

# torch.Tensor.gather
src_logits.gather(2, activated_class_ids)

# third_party package
from torchvision import models

models.resnet50

import mmcv
import numpy

mmcv.load
mmcv.dump

import mmdet as det

det.models.build_backbone

import mmdet3d.core
from mmdet3d import core

core.bbox.structures.lidar_box3d.LiDARInstance3DBoxes
mmdet3d.core.draw_heatmap_gaussian

# delete api
x.cuda()
a == x.is_contiguous()
torch.backends.cudnn.deterministic()

# torch.Tensor.bfloat16/bool/byte/char/double/float
# torch.Tensor.half/int/long/short/chalf/cfloat/cdouble
x.bfloat16()
x.bool()
x.byte()
x.char()
x.double()
x.float()
x.half()
x.int()
x.long()
x.short()
x.chalf()
x.cfloat()
x.cdouble()

# torch.Tensor.expand
x.expand(2)
x.expand(2, 3)
x.expand([2])
x.expand((2, 3))
x.expand([2, 3])
x.expand(size=[2, 3])
x.expand(size=(2, 3))

list1 = [2, 3]
x.expand(list1)

list1 = (2, 3)
x.expand(*list1)

# torch.Tensor.masked_fill
mask = mask.float().masked_fill(mask == 1, float("-inf"))

# torch.nn.CrossEntropyLoss
torch.nn.CrossEntropyLoss(reduction="none")

# torch.tensor
## case 1: 指定 dtype、device、requires_grad、pin_memory
a = torch.tensor(
    torch.tensor([2, 3, 4]),
    dtype=torch.float32,
    device=torch.device("cuda"),
    requires_grad=True,
)
print("[torch.tensor case-1]: ", a.shape, a.dtype)

## Case2:
flag = True
a = torch.tensor(
    torch.tensor([2, 3, 4]),
    dtype=torch.float32,
    device=torch.device("cuda"),
    requires_grad=flag,
)
print("[torch.tensor case-2]: ", a.shape, a.dtype)

# torch.cuda.is_available
## Case1:
print("[torch.cuda.is_available case-1]: ", torch.cuda.is_available())

# torch.Tensor
## Case1:
def a(x: torch.Tensor):
    pass


## Case2:
a = torch.Tensor(2, 3)
print("[torch.Tensor case-2]: ", a.shape, a.dtype)

# torch.LongTensor
## Case1:
def a(x: torch.LongTensor):
    pass


## Case2:
a = torch.LongTensor(2, 3)
print("[LongTensor case-2]: ", a.shape, a.dtype)

# torch.IntTensor
## Case1:
def a(x: torch.IntTensor):
    pass


## Case2:
a = torch.IntTensor(2, 3, 6)
print("[IntTensor case-2]: ", a.shape, a.dtype)


# torch.FloatTensor
## Case1:
def a(x: torch.FloatTensor):
    pass


## Case2:
a = torch.FloatTensor(2, 3, 6)
print("[FloatTensor case-2]: ", a.shape, a.dtype)

# torch.nn.functional.interpolate
## Case1:
a = torch.nn.functional.interpolate(torch.randn(1, 2, 20, 20), [24, 24])
print("[nn.functional.interpolate case-1]: ", a.shape)

## Case2:
a = torch.nn.functional.interpolate(torch.rand(1, 2, 20, 20), scale_factor=0.6)
print("[nn.functional.interpolate case-2]: ", a.shape)

# torch.equal
## Case1:
r = torch.equal(torch.tensor([1, 2]), torch.tensor([1, 2]))
print("[equal]: ", r)

# torch.randint
## Case1:
a = torch.randint(2, 5, [3, 4], device=torch.device("cuda"))
print("[randint]: ", a.shape, a.min(), a.max())

## Case2:
torch.randint(10, [2, 2])
print("[randint]: ", a.shape, a.min(), a.max())

## Case3:
a, b = 2, 25
a = torch.randint(a, b, [3, 4], device=torch.device("cuda"))
print("[randint]: ", a.shape, a.min(), a.max())

# torch.__version__
## Case1:
print(torch.__version__)

# torch.Tensor.new_tensor
## Case1:
a = torch.tensor([1, 2, 3])
b = a.new_tensor([4, 5, 6], dtype=torch.float64, requires_grad=True)
print("[Tensor.new_tensor case-1]: ", b)

## Case2:
a = torch.tensor([1, 2, 3], dtype=torch.int64)
b = a.new_tensor([4, 5, 6])
print("[Tensor.new_tensor case-2]: ", b)

# torch.Tensor.new_zeros
## Case1:
a = torch.tensor([1, 2, 3], dtype=torch.int64)
b = a.new_zeros([3, 4], dtype=torch.float64, requires_grad=True)
print("[Tensor.new_zeros case-1]: ", b)

## Case2:
a = torch.tensor([1, 2, 3], dtype=torch.int64)
b = a.new_zeros(3, 4)
print("[Tensor.new_zeros case-2]: ", b)

## Case3:
b = a.new_zeros([3, 4])
print("[Tensor.new_zeros case-3]: ", b)

# torch.Tensor.new_ones
## Case1:
a = torch.tensor([1, 2, 3], dtype=torch.int64)
b = a.new_ones([3, 4], dtype=torch.float64, requires_grad=True, pin_memory=True)
print("[Tensor.new_ones case-1]: ", b)

## Case2:
a = torch.tensor([1, 2, 3], dtype=torch.float64)
b = a.new_ones(3, 4, requires_grad=True)
print("[Tensor.new_ones case-2]: ", b)

## Case3:
b = a.new_ones([3, 4])
print("[Tensor.new_ones case-3]: ", b)

# torch.Tensor.new_full
## Case1:
a = torch.tensor([1, 2, 3], dtype=torch.int64)
b = a.new_full([3, 4], 2.43, dtype=torch.float64, requires_grad=True, pin_memory=True)
print("[Tensor.new_full case-1]: ", b)

## Case2:
flag = False
a = torch.tensor([1, 2, 3], dtype=torch.int64)
b = a.new_full((2, 3), 4, requires_grad=flag)
print("[Tensor.new_full case-2]: ", b)

# torch.new_empty
## Case1:
a = torch.tensor([1, 2, 3], dtype=torch.int64)
b = a.new_empty((3, 4), dtype=torch.float64, requires_grad=True, pin_memory=True)
print("[Tensor.new_empty case-1]: ", b)

# torch.Tensor.normal_
## Case1:
a = torch.tensor([1, 3, 4, 9, 0.5, 1.5])
a = a.normal_(0.2, 0.3)
print("[Tensor.normal_ case-1]: ", a)


# torch.Tensor.uniform_
## Case1:
c = torch.tensor(a.uniform_(2, 6))
print("[Tensor.uniform_ case-1]: ", c)

# torch.Tensor.expand
## Case1:
x = torch.tensor([[1], [2], [3]])
y = x.expand(3, 4)
print("[Tensor.expand case-1]: ", y.shape)

## Case2:
x = torch.tensor([[1], [2], [3]])
y = x.expand((3, 4))
print("[Tensor.expand case-2]: ", y.shape)

torch.random.manual_seed(23)

# torch.Tensor.new_zeros/new_ones/new_empty
x.new_zeros(2)

x.new_zeros(2, 3)

x.new_zeros([2, 3])

x.new_zeros((2, 3))

## corner case
shape = 2
x.new_zeros(shape)

shape = (2, 3)
x.new_zeros(shape, requires_grad=True)

shape = (2, 3)
x.new_zeros(*shape, requires_grad=True, dtype=torch.float32)

x.new_zeros(*shape, requires_grad=True, dtype=torch.float32, pin_memory=True)

x.new_zeros(*shape, requires_grad=True, dtype=torch.float32, pin_memory=False)

x.new_zeros(*x.size())

x.new_zeros(x.size())

# torch.Tensor.new_full

x.new_full([2, 3], 2.0)

x.new_full([2, 3], 2.0, requires_grad=True)

x.new_full([2, 3], 2.0, requires_grad=True, pin_memory=False)

x.new_full([2, 3], 2.0, dtype=torch.float32, requires_grad=True, pin_memory=True)


import torch

# torch.Generator()
## Case1: default cpu
g_cpu = torch.Generator()
print("[torch.Generator() case-1]: ", g_cpu)

## Case2: cpu
g_cpu = torch.Generator(device="cpu")
print("[torch.Generator() case-2]: ", g_cpu)

## Case3: cpu
g_cpu = torch.Generator("cpu")
print("[torch.Generator() case-3]: ", g_cpu)

# case 1:
print(torch.Size([2, 8, 64, 64]))

# # case 2:
assert torch.randn(6, 5, 7).size() == torch.Size([6, 5, 7])

# # case 3:
out = torch.Size([6, 5, 7])
shape_nchw = torch.Size([6, 5, 7])
assert out == torch.Size(shape_nchw)

# case 4:
print(torch.Size([1]))

# case 5
shape = torch.Size([1])

# torch.Tensor.index_copy_
## case 1: index axis = 0
import torch

x = torch.zeros(5, 3)
t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
index = torch.tensor([0, 4, 2])
x.index_copy_(0, index, t)
print("[torch.Tensor.index_copy_ case-1]: ", x)

## Case2:  index axis !=0  high dimension data that x and tensor must have same shape
x = torch.zeros(2, 1, 3, 3)
t = torch.tensor(
    [[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]],
    dtype=torch.float,
)
index = torch.tensor([0, 1, 2])
x.index_copy_(2, index, t)
print("[torch.Tensor.index_copy_ case-2]: ", x)

## case 3: assign case 1
x = torch.zeros(5, 3)
t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
index = torch.tensor([0, 4, 2])
y = x.index_copy_(0, index, t)
print("[torch.Tensor.index_copy_ case-3]: ", y)

## Case4: assign case 2
x = torch.zeros(2, 1, 3, 3)
t = torch.tensor(
    [[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]],
    dtype=torch.float,
)
index = torch.tensor([0, 1, 2])
y = x.index_copy_(2, index, t)
print("[torch.Tensor.index_copy_ case-4]: ", y)


## Case5: scalar and assign
x = torch.zeros(20)
t = torch.tensor([1, 3, 4, 5], dtype=torch.float)
index = torch.tensor([0, 12, 2, 1])
y = x.index_copy_(0, index, t)
print("[torch.Tensor.index_copy_ case-5]: ", y)

import torch

data = torch.tensor([23.0, 32.0, 43.0])
# case 1:
if not data.requires_grad:
    print(1)

# case 2:
print(data.requires_grad)

# case 3:
data.requires_grad = False

# case 4:
requires_grad = data.requires_grad

# # case 5
data = torch.tensor([23.0, 32.0, 43.0], requires_grad=data.requires_grad)

# case 6
print(data.requires_grad == False)

# case 7:
print(not data.requires_grad)

# case 8:
print("{} , {}".format("1", str(data.requires_grad)))

# case 9:
def test():
    return True


data.requires_grad = test()

# case 10:
z = (True, False, True)
a, data.requires_grad, c = z
print(data.requires_grad)

import torch
import torch.nn as nn

# case 1
m = nn.InstanceNorm3d(100)
input = torch.randn(20, 100, 35, 45, 10)
output = m(input)
print("[torch.nn.InstanceNorm3d case-1]: ", output)

# case 2
m = nn.InstanceNorm3d(100, affine=True)
input = torch.randn(20, 100, 35, 45, 10)
output = m(input)
print("[torch.nn.InstanceNorm3d case-2]: ", output)

# case 3
m = nn.InstanceNorm3d(100, affine=False)
input = torch.randn(20, 100, 35, 45, 10)
output = m(input)
print("[torch.nn.InstanceNorm3d case-3]: ", output)

# case 4
m = nn.InstanceNorm3d(100, affine=True, momentum=0.1)
input = torch.randn(20, 100, 35, 45, 10)
output = m(input)
print("[torch.nn.InstanceNorm3d case-4]: ", output)

# case 5
m = nn.InstanceNorm3d(100, affine=False, momentum=0.1)
input = torch.randn(20, 100, 35, 45, 10)
output = m(input)
print("[torch.nn.InstanceNorm3d case-5]: ", output)

import torch
import torch.nn as nn

# case 1
loss = torch.nn.BCEWithLogitsLoss(reduction="none")
input = torch.tensor([1.0, 0.7, 0.2], requires_grad=True)
target = torch.tensor([1.0, 0.0, 0.0])
output = loss(input, target)
print("[torch.nn.BCEWithLogitsLoss case-1]: ", output)

# case 2
loss = nn.BCEWithLogitsLoss(weight=torch.tensor([1.0, 0.2, 0.2]), reduction="none")
input = torch.tensor([1.0, 0.7, 0.2], requires_grad=True)
target = torch.tensor([1.0, 0.0, 0.0])
output = loss(input, target)
print("[torch.nn.BCEWithLogitsLoss case-2]: ", output)

# case 3
loss = nn.BCEWithLogitsLoss(pos_weight=torch.ones([3]))
input = torch.tensor([1.0, 0.7, 0.2], requires_grad=True)
target = torch.tensor([1.0, 0.0, 0.0])
output = loss(input, target)
print("[torch.nn.BCEWithLogitsLoss case-3]: ", output)

# case 4
loss = nn.BCEWithLogitsLoss(size_average=True)
input = torch.tensor([1.0, 0.7, 0.2], requires_grad=True)
target = torch.tensor([1.0, 0.0, 0.0])
output = loss(input, target)
print("[torch.nn.BCEWithLogitsLoss case-4]: ", output)

# case 5
loss = nn.BCEWithLogitsLoss()
input = torch.tensor([1.0, 0.7, 0.2], requires_grad=True)
target = torch.tensor([1.0, 0.0, 0.0])
output = loss(input, target)
print("[torch.nn.BCEWithLogitsLoss case-5]: ", output)

import torch
import torch.nn as nn
from torch.utils.data import BatchSampler

# case 1
o = list(BatchSampler(range(10), batch_size=3, drop_last=True))
print("[torch.utils.data.BatchSampler case-1]: ", o)

# case 2
o = list(BatchSampler(range(10), batch_size=3, drop_last=False))
print("[torch.utils.data.BatchSampler case-2]: ", o)

# case 3
batch_sampler_train = torch.utils.data.BatchSampler(range(10), 2, drop_last=True)
print("[torch.utils.data.BatchSampler case-3]: ", list(batch_sampler_train))

# case 4
batch_size = 4
batch_sampler_train = torch.utils.data.BatchSampler(
    range(10), batch_size, drop_last=False
)
print("[torch.utils.data.BatchSampler case-4]: ", list(batch_sampler_train))

# case 5
batch_size = 4
batch_sampler_train = torch.utils.data.BatchSampler(
    sampler=range(10), batch_size=batch_size, drop_last=False
)
print("[torch.utils.data.BatchSampler case-5]: ", list(batch_sampler_train))

import torch

cpu = torch.device("cpu")
a = torch.randn(2, 3)
c = torch.randn(2, 3, dtype=torch.float64, device=cpu)
# torch.Tensor.to
## Case1:  torch.to(device=None, dtype=None, non_blocking=False, copy=False, memory_format=torch.preserve_format)
### torch code
b = a.to(cpu, non_blocking=False, copy=False)
print("[torch.Tensor.to case-1]: ", b)

## Case2:  torch.to(device=None, dtype=None, non_blocking=False, copy=False, memory_format=torch.preserve_format)
### torch code
b = a.to("cpu")
print("[torch.Tensor.to case-2]: ", b)

## Case3:  torch.to(device=None, dtype=None, non_blocking=False, copy=False, memory_format=torch.preserve_format)
### torch code
b = a.to(device=cpu, dtype=torch.float64)
print("[torch.Tensor.to case-3]: ", b)

## Case4:  torch.to(device=None, dtype=None, non_blocking=False, copy=False, memory_format=torch.preserve_format)
### to(dtype, non_blocking=False, copy=False, memory_format=torch.preserve_format)
### torch code
b = a.to(torch.float64)
print("[torch.Tensor.to case-4]: ", b)

## Case5:  torch.to(dtype, non_blocking=False, copy=False, memory_format=torch.preserve_format)
### torch code
b = a.to(dtype=torch.float64)
print("[torch.Tensor.to case-5]: ", b)


## Case6:  torch.to(other, non_blocking=False, copy=False)
### torch code
b = a.to(c)
print("[torch.Tensor.to case-6]: ", b)

## Case7:
# 不支持torch.channels_last
# a =troch.tensor([[[[1]]]])
# a = a.to(memory_format=torch.channels_last)
# print('[torch.Tensor.to case-7]: ', b)

## Case8:
a = a.to(torch.half)
print("[torch.Tensor.to case-8]: ", b)

## Case9:
table = a
b = a.to(table.device)
print("[torch.Tensor.to case-9]: ", b)

## Case10:
b = a.to(torch.float32)
print("[torch.Tensor.to case-10]: ", b)

## Case11:
device = torch.device("cpu")
b = torch.tensor([-1]).to(torch.bool)
print("[torch.Tensor.to case-11]: ", b)

## Case12:
dtype = torch.float32
b = a.to(dtype=dtype)
print("[torch.Tensor.to case-12]: ", b)

## Case13:
b = a.to(torch.device("cpu"))
print("[torch.Tensor.to case-13: ", b)
