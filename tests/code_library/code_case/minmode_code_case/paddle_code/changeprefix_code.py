import copy
import math
import pickle
from collections import OrderedDict

import numpy
import numpy as np
import paddle
import paddleformers
import setuptools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import transformers
from setuptools import setup
from torch.autograd import Function
from torch.distributed import ReduceOp
from torch.nn.attention import SDPBackend
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CppExtension
from torch.utils.data import (ChainDataset, ConcatDataset, Dataset,
                              IterableDataset, Sampler, SequentialSampler,
                              Subset, default_collate, get_worker_info)

############################## 相关utils函数，如下 ##############################

def device2str(type=None, index=None, *, device=None):
    type = device if device else type
    if isinstance(type, int):
        type = f'gpu:{type}'
    elif isinstance(type, str):
        if 'cuda' in type:
            type = type.replace('cuda', 'gpu')
        if 'cpu' in type:
            type = 'cpu'
        elif index is not None:
            type = f'{type}:{index}'
    elif isinstance(type, paddle.CPUPlace) or (type is None):
        type = 'cpu'
    elif isinstance(type, paddle.CUDAPlace):
        type = f'gpu:{type.get_device_id()}'

    return type
############################## 相关utils函数，如上 ##############################

result = torch.BFloat16Tensor([1.5, 2, 3])
result = torch.BFloat16Tensor()
result = torch.BoolTensor(2, 3)
shape = [2, 3]
result = torch.BoolTensor(*shape)
result = torch.ByteTensor(2, 3)
shape = [2, 3]
result = torch.ByteTensor(*shape)
result = torch.CharTensor(2, 3)
shape = [2, 3]
result = torch.CharTensor(*shape)
result = torch.DoubleTensor(2, 3)
shape = [2, 3]
result = torch.DoubleTensor(*shape)
result = torch.FloatTensor(2, 3)
shape = [2, 3]
result = torch.FloatTensor(*shape)
result = torch.Generator(device='cpu')
result = torch.Generator()
result = torch.HalfTensor(2, 3)
shape = [2, 3]
result = torch.HalfTensor(*shape)
result = torch.IntTensor(2, 3)
shape = [2, 3]
result = torch.IntTensor(*shape)
result = torch.LongTensor(2, 3)
shape = [2, 3]
result = torch.LongTensor(*shape)
result = torch.ShortTensor(2, 3)
shape = [2, 3]
result = torch.ShortTensor(*shape)
result = list(torch.Size([2, 8, 64, 64]))
result = torch.randn(6, 5, 7).size() == torch.Size([6, 5, 7])
result = torch.Tensor(2, 3)
shape = [2, 3]
result = torch.Tensor(*shape)
x = torch.arange(16).reshape(4, 4)
result = x.T
result = torch.arange(16).reshape(4, 4).T
x = torch.tensor([1, 2, 3], dtype=torch.int64)
y = torch.tensor([3, 2, 1], dtype=torch.int64)
result = x + y
x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
y = torch.tensor([3.0, 2.0, 1.0], dtype=torch.float32)
result = x + y
x = torch.tensor([True, False, True])
y = torch.tensor([True, True, False])
result = x & y
x = torch.tensor([1, 2, 3], dtype=torch.int32)
y = torch.tensor([3, 2, 1], dtype=torch.int32)
result = x & y
x = torch.tensor([1.0, 2.0, 3.0])
result = x.__array__()
x = torch.tensor([[1, 2], [3, 4]])
result = x.__array__()
x = torch.tensor([True])
result = bool(x)
x = torch.tensor([0.0])
result = bool(x)
x = torch.tensor([True])
result = copy.deepcopy(x)
x = torch.tensor([0.0])
result = copy.deepcopy(x)
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([1.0, 2.0, 3.0])
result = x == y
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([1.0, 5.0, 3.0])
result = x == y
x = torch.tensor([2.0, 4.0])
result = 8 // x
x = torch.tensor([1, 2])
result = 10 // x
x = torch.tensor(3.14159)
result = format(x, '.2f')
x = torch.tensor(123)
result = format(x, '05d')
x = torch.tensor([3.0, 2.0, 1.0])
y = torch.tensor([1.0, 2.0, 3.0])
result = x >= y
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([1.0, 5.0, -1.0])
result = x >= y
x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
result = x[1, 2]
x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
result = x[:, 1:3]
x = torch.tensor([3.0, 2.0, 1.0])
y = torch.tensor([1.0, 2.0, 3.0])
result = x > y
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([1.0, 5.0, -1.0])
result = x > y
x = torch.tensor(5)
result = x.__index__()
x = torch.tensor(3, dtype=torch.int64)
result = x.__index__()
x = torch.tensor(2.0)
result = int(x)
x = torch.tensor([2.0])
result = int(x)
x = torch.tensor([True, False])
result = ~x
x = torch.tensor([1, 2], dtype=torch.int32)
result = ~x
x = torch.tensor([True, False, True])
y = torch.tensor([True, True, False])
x |= y
x = torch.tensor([1, 2, 3], dtype=torch.int32)
y = torch.tensor([3, 2, 1], dtype=torch.int32)
x |= y
x = torch.tensor([3.0, 2.0, 1.0])
y = torch.tensor([1.0, 2.0, 3.0])
result = x <= y
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([1.0, 5.0, -1.0])
result = x <= y
x = torch.tensor([1.0, 2.0, 3.0, 4.0])
result = len(x)
x = torch.tensor([[1, 2], [3, 4], [5, 6]])
result = len(x)
x = torch.tensor([3.0, 2.0, 1.0])
y = torch.tensor([1.0, 2.0, 3.0])
result = x < y
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([1.0, 5.0, -1.0])
result = x < y
x = torch.tensor([1, 2, 3], dtype=torch.int64)
y = torch.tensor([3, 2, 1], dtype=torch.int64)
result = x * y
x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
y = torch.tensor([3.0, 2.0, 1.0], dtype=torch.float32)
result = x * y
x = torch.tensor([3.0, 2.0, 1.0])
y = torch.tensor([1.0, 2.0, 3.0])
result = x != y
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([1.0, 5.0, -1.0])
result = x != y
x = torch.tensor([1.0, 2.0], dtype=torch.float32)
result = -x
x = torch.tensor([3.0, -1.0], dtype=torch.float64)
result = -x
x = torch.tensor([1.0], dtype=torch.float32)
result = not x
x = torch.tensor([3.0], dtype=torch.float64)
result = not x
x = torch.tensor([True, False, True])
y = torch.tensor([True, True, False])
result = x | y
x = torch.tensor([1, 2, 3], dtype=torch.int32)
y = torch.tensor([3, 2, 1], dtype=torch.int32)
result = x | y
x = torch.tensor([2.0, 3.0])
result = x ** 2
x = torch.tensor([1.0, 2.0])
result = x ** 3.0
x = torch.tensor([1, 2, 3], dtype=torch.int64)
result = 5 + x
x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
result = 5.0 + x
x = torch.tensor([1, 2, 3], device=torch.device('cpu'), dtype=torch.int64,
    requires_grad=False)
data = pickle.dumps(x)
result = pickle.loads(data)
x = torch.tensor([1, 2, 3], device=torch.device('cuda'), dtype=torch.int64,
    pin_memory=False, requires_grad=False)
data = pickle.dumps(x)
result = pickle.loads(data)
x = torch.tensor([1, 2, 3], dtype=torch.int64)
result = 5 * x
x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
result = 5.0 * x
x = torch.tensor([2.0, 3.0])
result = 2 ** x
x = torch.tensor([1.0, 2.0])
result = 3.0 ** x
x = torch.tensor([2.0, 3.0])
result = 5 - x
x = torch.tensor([1, 2])
result = 10 - x
x = torch.tensor([2.0, 4.0])
result = 8 / x
x = torch.tensor([1, 2])
result = 10 / x
x = torch.tensor([1.0, 2.0, 3.0])
x[1] = 5.0
result = x
x = torch.tensor([[1, 2], [3, 4]])
x[0, :] = torch.tensor([5, 6])
result = x
x = torch.tensor([1, 2, 3], dtype=torch.int64)
y = torch.tensor([3, 2, 1], dtype=torch.int64)
result = x - y
x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
y = torch.tensor([3.0, 2.0, 1.0], dtype=torch.float32)
result = x - y
x = torch.tensor([True, False, True])
y = torch.tensor([True, True, False])
result = x ^ y
x = torch.tensor([1, 2, 3], dtype=torch.int32)
y = torch.tensor([3, 2, 1], dtype=torch.int32)
result = x ^ y
a = torch.tensor([[-4, 9], [-23, 2]])
result = a.abs()
result = torch.tensor([[-4, 9], [-23, 2]]).abs()
a = torch.tensor([-1])
a.abs_()
a = torch.tensor([-1, -2, 3])
a.abs_()
a = torch.tensor([[0.3348, -0.5889, 0.2005, -0.1584], [0.3348, -0.5889, 
    0.2005, -0.1584]])
result = a.acos()
result = torch.tensor([[0.3348, -0.5889, 0.2005, -0.1584]]).acos()
a = torch.tensor([0.34, -0.56, 0.73])
a.acos_()
a = torch.tensor([1.0, -1.0, 0.0])
a.acos_()
result = torch.tensor([1.3192, 1.9915, 1.9674, 1.7151]).acosh()
a = torch.tensor([1.3192, 1.9915, 1.9674, 1.7151])
result = a.acosh()
result = torch.tensor([1.3192, 1.9915, 1.9674, 1.7151]).acosh_()
a = torch.tensor([1.3192, 1.9915, 1.9674, 1.7151])
result = a.acosh_()
x = torch.tensor([1, 2, 3])
result = x.add(torch.tensor([1, 4, 6]))
x = torch.tensor([1, 2, 3])
result = x.add(20)
x = torch.tensor([1, 2, 3])
x.add_(torch.tensor([1, 4, 6]))
x = torch.tensor([1, 2, 3])
x.add_(20)
x = torch.tensor([[1, 2], [4, 5]])
mat1 = torch.tensor([[1, 2], [4, 5]])
mat2 = torch.tensor([[1, 2], [4, 5]])
result = x.addmm(mat1, mat2)
x = torch.tensor([[1.0, 2], [4, 5]])
mat1 = torch.tensor([[1.0, 2], [4, 5]])
mat2 = torch.tensor([[1.0, 2], [4, 5]])
result = x.addmm(mat1, mat2, beta=0.6, alpha=0.7)
x = torch.tensor([[1, 2], [4, 5]])
mat1 = torch.tensor([[1, 2], [4, 5]])
mat2 = torch.tensor([[1, 2], [4, 5]])
x.addmm_(mat1, mat2)
x = torch.tensor([[1.0, 2], [4, 5]])
mat1 = torch.tensor([[1.0, 2], [4, 5]])
mat2 = torch.tensor([[1.0, 2], [4, 5]])
x.addmm_(mat1, mat2, beta=0.6, alpha=0.7)
a = torch.rand(1, 2).bool()
result = a.all()
a = torch.rand(3, 4)
result = a.all()
x = torch.tensor([[1, 2, 3], [3, 4, 6]])
result = x.amax()
x = torch.tensor([[1, 2, 3], [3, 4, 6]])
result = x.amax(dim=1)
x = torch.tensor([[1, 2, 3], [3, 4, 6]])
result = x.amin()
x = torch.tensor([[1, 2, 3], [3, 4, 6]])
result = x.amin(dim=1)
result = torch.tensor([-1 + 1.0j, -2 + 2.0j, 3 - 3.0j]).angle()
x = torch.tensor([-1 + 1.0j, -2 + 2.0j, 3 - 3.0j])
result = x.angle() * 180 / 3.14159
a = torch.rand(1, 2).bool()
result = a.any()
a = torch.rand(3, 4)
result = a.any()
x = torch.tensor([1.3192, 1.9915, 1.9674, 1.7151])
result = x.apply_(lambda x: x * 2)
x = torch.tensor([[1, 2, 3], [3, 4, 6]])
result = x.argmax()
x = torch.tensor([[1, 2, 3], [3, 4, 6]])
result = x.argmax(dim=1)
x = torch.tensor([[1, 2, 3], [3, 4, 6]])
result = x.argmin()
x = torch.tensor([[1, 2, 3], [3, 4, 6]])
result = x.argmin(dim=1)
input = torch.tensor([[1.3398, 0.2663, -0.2686, 0.245], [-0.7401, -0.8805, 
    -0.3402, -1.1936], [0.4907, -1.3948, -1.0691, -0.3132], [-1.6092, 
    0.5419, -0.2993, 0.3195]])
result = input.argsort()
input = torch.tensor([[1.3398, 0.2663, -0.2686, 0.245], [-0.7401, -0.8805, 
    -0.3402, -1.1936], [0.4907, -1.3948, -1.0691, -0.3132], [-1.6092, 
    0.5419, -0.2993, 0.3195]])
result = input.argsort(dim=1)
x = torch.tensor([[0.0335, 0.183, -0.1269], [0.1897, -0.1422, -0.494], [-
    0.7674, -0.0134, -0.3733]])
results = x.as_strided((2, 2), (1, 2))
x = torch.tensor([[0.0335, 0.183, -0.1269], [0.1897, -0.1422, -0.494], [-
    0.7674, -0.0134, -0.3733]])
results = x.as_strided((2, 2), (1, 2), 0)
result = torch.tensor([0.34, -0.56, 0.73]).asin()
a = torch.tensor([0.34, -0.56, 0.73])
result = a.asin()
result = torch.tensor([0.34, -0.56, 0.73]).asin_()
a = torch.tensor([0.34, -0.56, 0.73])
result = a.asin_()
a = torch.Tensor([[1.0, 2.0], [3.0, 4.0]])
result = a.asinh()
result = torch.tensor([0.34, -0.56, 0.73]).asinh_()
a = torch.tensor([0.34, -0.56, 0.73])
result = a.asinh_()
result = torch.tensor([0.34, -0.56, 0.73]).atan()
a = torch.tensor([0.34, -0.56, 0.73])
result = a.atan()
input = torch.tensor([0.9041, 0.0196, -0.3108, -2.4423])
other = torch.tensor([0.2341, 0.2539, -0.6256, -0.6448])
result = input.atan2(other)
>>>>>>result = torch.tensor([0.9041, 0.0196, -0.3108, -2.4423]).atan2(torch.
    tensor([0.2341, 0.2539, -0.6256, -0.6448]))
result = torch.tensor([0.34, -0.56, 0.73]).atan_()
a = torch.tensor([0.34, -0.56, 0.73])
result = a.atan_()
a = torch.Tensor([[1.0, 2.0], [3.0, 4.0]])
result = a.atanh()
result = torch.tensor([0.34, -0.56, 0.73]).atanh_()
a = torch.tensor([0.34, -0.56, 0.73])
result = a.atanh_()
a = torch.tensor([[[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]]])
b = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
input = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
result = input.baddbmm(a, b)
a = torch.tensor([[[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]]])
b = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
input = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
result = input.baddbmm(a, b, beta=3)
a = torch.tensor([[[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]]])
b = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
input = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
result = input.baddbmm_(a, b)
a = torch.tensor([[[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]]])
b = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
input = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
result = input.baddbmm_(a, b, beta=3)
input = torch.tensor([4, 3, 6, 3, 4])
weights = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
result = input.bincount()
input = torch.tensor([4, 3, 6, 3, 4])
weights = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
result = input.bincount(weights)
x = torch.tensor([1, 2, 3])
y = torch.tensor([-1, 9, 10])
result = x.bitwise_and(y)
x = torch.tensor([True, False, True])
y = torch.tensor([False, False, True])
result = x.bitwise_and(y)
x = torch.tensor([1, 2, 3])
y = torch.tensor([-1, 9, 10])
x.bitwise_and_(y)
x = torch.tensor([True, False, True])
y = torch.tensor([False, False, True])
x.bitwise_and_(y)
input = torch.tensor([-2, -7, 31], dtype=torch.int8)
other = torch.tensor([1, 0, 3], dtype=torch.int8)
result = input.bitwise_left_shift(other)
input = torch.tensor([-2, -7, 31], dtype=torch.int8)
other = torch.tensor([1, 0, 3], dtype=torch.int8)
result = input.bitwise_left_shift(other=other)
input = torch.tensor([-2, -7, 31], dtype=torch.int8)
other = torch.tensor([1, 0, 3], dtype=torch.int8)
result = input.bitwise_left_shift_(other)
input = torch.tensor([-2, -7, 31], dtype=torch.int8)
other = torch.tensor([1, 0, 3], dtype=torch.int8)
result = input.bitwise_left_shift_(other=other)
x = torch.tensor([1, 2, 3])
result = x.bitwise_not()
x = torch.tensor([True, False, True])
result = x.bitwise_not()
x = torch.tensor([1, 2, 3])
x.bitwise_not_()
x = torch.tensor([True, False, True])
x.bitwise_not_()
x = torch.tensor([1, 2, 3])
y = torch.tensor([-1, 9, 10])
result = x.bitwise_or(y)
x = torch.tensor([True, False, True])
y = torch.tensor([False, False, True])
result = x.bitwise_or(y)
x = torch.tensor([1, 2, 3])
y = torch.tensor([-1, 9, 10])
x.bitwise_or_(y)
x = torch.tensor([True, False, True])
y = torch.tensor([False, False, True])
x.bitwise_or_(y)
input = torch.tensor([-2, -7, 31], dtype=torch.int8)
other = torch.tensor([1, 0, 3], dtype=torch.int8)
result = input.bitwise_right_shift(other)
input = torch.tensor([-2, -7, 31], dtype=torch.int8)
other = torch.tensor([1, 0, 3], dtype=torch.int8)
result = input.bitwise_right_shift(other=other)
input = torch.tensor([-2, -7, 31], dtype=torch.int8)
other = torch.tensor([1, 0, 3], dtype=torch.int8)
result = input.bitwise_right_shift_(other)
input = torch.tensor([-2, -7, 31], dtype=torch.int8)
other = torch.tensor([1, 0, 3], dtype=torch.int8)
result = input.bitwise_right_shift_(other=other)
x = torch.tensor([1, 2, 3])
y = torch.tensor([-1, 9, 10])
result = x.bitwise_xor(y)
x = torch.tensor([True, False, True])
y = torch.tensor([False, False, True])
result = x.bitwise_xor(y)
x = torch.tensor([1, 2, 3])
y = torch.tensor([-1, 9, 10])
x.bitwise_xor_(y)
x = torch.tensor([True, False, True])
y = torch.tensor([False, False, True])
x.bitwise_xor_(y)
a = torch.tensor([[[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]]])
b = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
result = a.bmm(b)
>>>>>>result = torch.tensor([[[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]]]).bmm(torch.
    tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]))
x = torch.tensor([1, 2, 3])
result = x.broadcast_to((3, 3))
x = torch.tensor([1, 2, 3])
shape = [3, 3]
result = x.broadcast_to(size=shape)
x = torch.randn([3, 4])
result = x.cauchy_()
x = torch.randn([3, 4])
result = x.cauchy_(1.0, 2.0)
a = torch.tensor([1.1, 2.5, 3.6, 4.8])
result = a.ceil()
result = torch.tensor([0.1, -1.5, -2.3, 3.8]).ceil()
result = torch.tensor([-0.6341, -1.4208, -1.09, 0.5826]).ceil_()
input = torch.tensor([-0.6341, -1.4208, -1.09, 0.5826])
result = input.ceil_()
x = torch.tensor([[2.4112, -0.7486, 1.4551], [-0.7486, 1.3544, 0.1294], [
    1.4551, 0.1294, 1.6724]])
result = x.cholesky()
x = torch.tensor([[2.4112, -0.7486, 1.4551], [-0.7486, 1.3544, 0.1294], [
    1.4551, 0.1294, 1.6724]])
result = x.cholesky(True)
a = torch.tensor([[0.9967, 0.0, 0.0], [-0.6374, 0.686, 0.0], [1.5858, -
    1.0314, 2.6615]])
result = a.cholesky_inverse()
a = torch.tensor([[0.9967, -0.6374, 1.5858], [0.0, 0.686, -1.0314], [0.0, 
    0.0, 2.6615]])
result = a.cholesky_inverse(upper=True)
x = torch.ones(2, 3)
result = x.chunk(2)
result = torch.ones(2, 3).chunk(chunks=2)
a = torch.tensor([-1.712, 0.1734, -0.0478, 0.8922])
result = a.clamp(-0.5, 0.5)
a = torch.tensor([-1.712, 0.1734, -0.0478, 0.8922])
result = a.clamp(min=-0.2, max=0.5)
a = torch.tensor([-1.712, 0.1734, -0.0478, 0.8922])
result = a.clamp_(-0.5, 0.5)
a = torch.tensor([-1.712, 0.1734, -0.0478, 0.8922])
result = a.clamp_(min=-0.2, max=0.5)
x = torch.tensor([-1.712, 0.1734, -0.0478, -0.0922])
result = x.clip(-0.5, 0.5)
x = torch.tensor([-1.712, 0.1734, -0.0478, -0.0922])
min, max = -0.5, 0.5
result = x.clip(min, max)
x = torch.tensor([-1.712, 0.1734, -0.0478, -0.0922])
result = x.clip_(-0.5, 0.5)
x = torch.tensor([-1.712, 0.1734, -0.0478, -0.0922])
min, max = -0.5, 0.5
result = x.clip_(min, max)
x = torch.tensor([1, 2, 3])
result = x.clone()
result = torch.tensor([1, 2, 3]).clone()
i = torch.tensor([[0, 1, 1], [2, 0, 2]])
v = torch.tensor([3, 4, 5], dtype=torch.float32)
x = paddle.sparse.sparse_coo_tensor(indices=i, values=v, shape=[2, 4])
result = x.coalesce()
result = result.to_dense()
src = torch.tensor([-1 + 1.0j, -2 + 2.0j, 3 - 3.0j])
result = src.conj()
src = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
result = src.contiguous()
src = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
>>>>>>result = src.contiguous(memory_format=torch.contiguous_format)
src = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
dst = torch.tensor([2.0, 2.0, 3.0, 4.0, 5.0, 6.0])
result = src.copy_(dst)
src = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
dst = torch.tensor([2.0, 2.0, 3.0, 4.0, 5.0, 6.0])
result = src.copy_(dst, non_blocking=False)
a = torch.tensor([-1.2557, -0.0026, -0.5387, 0.474, -0.9244])
result = a.copysign(1.0)
a = torch.tensor([[0.7079, 0.2778, -1.0249, 0.5719], [-0.0059, -0.26, -
    0.4475, -1.3948], [0.3667, -0.9567, -2.5757, -0.1751], [0.2046, -0.0742,
    0.2998, -0.1054]])
b = torch.tensor([0.2373, 0.312, 0.319, -1.1128])
result = a.copysign(b)
a = torch.tensor([-1.2557, -0.0026, -0.5387, 0.474, -0.9244])
result = a.copysign_(1.0)
a = torch.tensor([[0.7079, 0.2778, -1.0249, 0.5719], [-0.0059, -0.26, -
    0.4475, -1.3948], [0.3667, -0.9567, -2.5757, -0.1751], [0.2046, -0.0742,
    0.2998, -0.1054]])
b = torch.tensor([0.2373, 0.312, 0.319, -1.1128])
result = a.copysign_(b)
x = torch.tensor([[0.7308, 1.006, 0.527, 1.4516], [-0.1383, 1.5706, 0.4724,
    0.4141], [0.1193, 0.2829, 0.9037, 0.3957], [-0.8202, -0.6474, -0.1631, 
    -0.6543]])
result = x.corrcoef()
x = torch.tensor([[-0.1533, 2.302, -0.1771, 0.5928], [0.4338, -0.6537, 
    0.2296, 0.5946], [-0.4932, 1.8386, -0.1039, 1.044], [0.1735, -0.8303, -
    0.3821, -0.4384], [-0.1533, 2.302, -0.1771, 0.5928], [0.4338, -0.6537, 
    0.2296, 0.5946], [-0.4932, 1.8386, -0.1039, 1.044], [0.1735, -0.8303, -
    0.3821, -0.4384]])
result = x.corrcoef()
result = torch.tensor([1.4309, 1.2706, -0.8562, 0.9796]).cos()
a = torch.tensor([1.4309, 1.2706, -0.8562, 0.9796])
result = a.cos()
a = torch.tensor([1.4309, 1.2706, -0.8562, 0.9796])
a.cos_()
result = torch.tensor([1.4309, 1.2706, -0.8562, 0.9796]).cosh()
a = torch.tensor([1.4309, 1.2706, -0.8562, 0.9796])
result = a.cosh()
a = torch.tensor([1.4309, 1.2706, -0.8562, 0.9796])
a.cosh_()
a = torch.tensor([1, 2, 3])
result = a.cpu()
a = torch.tensor([1, 2, 3])
result = a.T.cpu()
x = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
y = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
result = x.cross(y)
x = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
y = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
result = x.cross(y, 1)
x = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
result = x.cumprod(0)
x = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
result = x.cumprod(dim=1)
x = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
x.cumprod_(0)
x = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
x.cumprod_(dim=1)
x = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
result = x.cumsum(0)
x = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
result = x.cumsum(dim=1)
x = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
x.cumsum_(0)
x = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
x.cumsum_(dim=1)
x = torch.tensor([1.3192, 1.9915, 1.9674, 1.7151])
result = x.data
x = torch.tensor([1.3192, 1.9915, 1.9674, 1.7151])
x.data = torch.tensor([1.0, 1.0, 1.0, 1.0])
result = x.data
a = a = torch.tensor([[1, 2, 3], [1, 2, 3]])
result = a.data_ptr()
a = a = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
result = a.data_ptr()
x = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
result = x.deg2rad()
i = torch.tensor([[0, 1, 1], [2, 0, 2]])
v = torch.tensor([3, 4, 5], dtype=torch.float32)
x = paddle.sparse.sparse_coo_tensor(indices=i, values=v, shape=[2, 4])
result = x.dense_dim()
x = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]],
    requires_grad=True)
result = x.detach()
result = torch.tensor([[1.0, 1.0, 1.0]], requires_grad=True).detach()
x = torch.tensor([[1.0, 1.0, 1.0]], requires_grad=True)
x.detach_()
x = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]],
    requires_grad=True)
linear = paddle.compat.nn.Linear(3, 4, bias=False)
linear.weight.data.fill_(0.1)
y = linear(x)
y.detach_()
src = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).to('cuda')
result = src.device
src = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).to('cpu')
result = src.device
a = torch.tensor([0.595, -0.0872, 2.3298])
result = a.diag()
a = torch.tensor([0.595, -0.0872, 2.3298])
result = a.diag(diagonal=1)
x = torch.tensor([[0.7545889, -0.25074545, 0.5929117], [-0.6097662, -
    0.01753256, 0.619769]])
result = x.diag_embed()
x = torch.tensor([[0.7545889, -0.25074545, 0.5929117], [-0.6097662, -
    0.01753256, 0.619769]])
result = x.diag_embed(1)
a = torch.tensor([1, 2, 3])
result = a.diagflat()
a = torch.tensor([1, 2, 3])
result = a.diagflat(1)
x = torch.tensor([[0.7545889, -0.25074545, 0.5929117], [-0.6097662, -
    0.01753256, 0.619769]])
result = x.diagonal()
x = torch.tensor([[0.7545889, -0.25074545, 0.5929117], [-0.6097662, -
    0.01753256, 0.619769]])
result = x.diagonal(1)
x = torch.tensor([1, 3, 2])
result = x.diff()
x = torch.tensor([1, 3, 2])
b = torch.tensor([4, 5])
result = x.diff(append=b)
result = torch.tensor([1, 0.5]).digamma()
a = torch.tensor([1, 0.5])
result = a.digamma()
a = torch.tensor([1, 0.5])
a.digamma_()
result = torch.tensor([1, 0.5]).dim()
a = torch.tensor([[1, 0.5]])
result = a.dim()
input = torch.tensor([-1.5393, -0.8675, 0.5916, 1.6321])
other = torch.tensor([0.0967, -1.0511, 0.6295, 0.836])
result = input.dist(other, 2)
input = torch.tensor([-1.5393, -0.8675, 0.5916, 1.6321])
other = torch.tensor([0.0967, -1.0511, 0.6295, 0.836])
result = input.dist(other, p=2.5)
a = torch.tensor([0.595, -0.0872, 2.3298, -0.2972])
result = a.div(torch.tensor([0.5]))
a = torch.tensor([0.595, -0.0872, 2.3298, -0.2972])
result = a.div(0.5)
a = torch.tensor([0.595, -0.0872, 2.3298, -0.2972])
a.div_(torch.tensor([0.5]))
a = torch.tensor([[0.595, -0.0872], [2.3298, -0.2972]])
b = torch.tensor([0.1815, -1.0111])
a.div_(other=b)
a = torch.tensor([0.595, -0.0872, 2.3298, -0.2972])
result = a.divide(torch.tensor([0.5]))
a = torch.tensor([0.595, -0.0872, 2.3298, -0.2972])
result = a.divide(0.5)
a = torch.tensor([0.595, -0.0872, 2.3298, -0.2972])
a.divide_(torch.tensor([0.5]))
a = torch.tensor([[0.595, -0.0872], [2.3298, -0.2972]])
b = torch.tensor([0.1815, -1.0111])
a.divide_(other=b)
result = torch.tensor([2, 3]).dot(torch.tensor([2, 1]))
x = torch.tensor([2, 3])
y = torch.tensor([2, 1])
result = x.dot(y)
src = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
print(src.dtype)
result = torch.tensor([1, 2, 3], dtype=src.dtype)
x = torch.tensor([1, 3, 2])
result = x.element_size()
x = torch.tensor([1, 3, 2], dtype=torch.int64)
result = x.element_size()
result = torch.tensor([[1, 2], [3, 4]]).eq(torch.tensor([[1, 1], [4, 4]]))
input = torch.tensor([[1, 2], [3, 4]])
other = torch.tensor([[1, 1], [4, 4]])
result = input.eq(other)
result = torch.tensor([0, -1.0, 10.0]).erf()
a = torch.tensor([0, -1.0, 10.0])
result = a.erf()
result = torch.tensor([0, 0.5]).erfinv()
a = torch.tensor([0, 0.5])
result = a.erfinv()
result = torch.tensor([0, 0.5]).erfinv_()
a = torch.tensor([0, 0.5])
result = a.erfinv_()
result = torch.tensor([0.0, -2.0, 3.0]).exp()
a = torch.tensor([-1.0, -2.0, 3.0])
result = a.exp()
result = torch.tensor([0.0, -2.0, 3.0]).exp_()
a = torch.tensor([-1.0, -2.0, 3.0])
result = a.exp_()
a = torch.tensor([1, 2, 3])
result = a.expand(3, 3)
result = torch.tensor([1, 2, 3]).expand(3, -1)
x = torch.tensor([[1], [2], [3]])
y = torch.randn(3, 4)
result = x.expand_as(y)
y = torch.randn(3, 4)
result = torch.tensor([[1], [2], [3]]).expand_as(y)
a = torch.tensor([1.0, 2.0, -3.0, -4.0, 5.0])
result = a.expm1()
a = torch.tensor([[1.0, 2.0, -3.0, -4.0, 5.0], [1.0, 2.0, -3.0, -4.0, 5.0]])
result = 2 * a.expm1()
input = torch.rand([5, 9])
result = input.fill_(3)
input = torch.rand([5, 9])
result = input.fill_(value=3)
x = torch.tensor([[[3.4742, 0.5466, -0.8008, -0.9079], [3.4742, 0.5466, -
    0.8008, -0.9079]]])
result = x.flatten()
x = torch.tensor([[[3.4742, 0.5466, -0.8008, -0.9079], [3.4742, 0.5466, -
    0.8008, -0.9079]]])
result = x.flatten(1)
x = torch.tensor([[0, 1], [2, 3]])
result = x.flip((0, 1))
x = torch.tensor([[0, 1], [2, 3]])
result = x.flip([0, 1])
input = torch.tensor([-0.8166, 1.5308, -0.253, -0.2091])
result = input.floor()
result = torch.tensor([-0.8166, 1.5308, -0.253, -0.2091]).floor()
a = torch.tensor([1.1, 2.5, 3.6, 4.8])
result = a.floor_()
result = torch.tensor([0.1, -1.5, -2.3, 3.8])
result.floor_()
a = torch.tensor([4.0, 3.0])
b = torch.tensor([2.0, 2.0])
result = a.floor_divide(b)
result = torch.tensor([4.0, 3.0]).floor_divide(torch.tensor([2.0, 2.0]))
result = torch.tensor([[1, 2], [3, 4]]).fmax(torch.tensor([[1, 1], [4, 4]]))
input = torch.tensor([[1, 2], [3, 4]])
other = torch.tensor([[1, 1], [4, 4]])
result = input.fmax(other)
result = torch.tensor([[1, 2], [3, 4]]).fmin(torch.tensor([[1, 1], [4, 4]]))
input = torch.tensor([[1, 2], [3, 4]])
other = torch.tensor([[1, 1], [4, 4]])
result = input.fmin(other)
result = torch.tensor([1, 2.5, -3.2]).frac()
a = torch.tensor([1, 2.5, -3.2])
result = a.frac()
a = torch.tensor([1, 2.5, -3.2])
a.frac_()
x = torch.arange(9.0)
a, b = x.frexp()
a = torch.tensor([[1, 2], [3, 4]])
result = a.gather(1, torch.tensor([[0, 0], [1, 0]]))
result = torch.tensor([[1, 2], [3, 4]]).gather(1, torch.tensor([[0, 0], [1,
    0]]))
a = torch.tensor([5, 10, 15])
b = torch.tensor([3, 4, 5])
result = a.gcd(b)
a = torch.tensor([5, 10, 15])
b = torch.tensor([3, 4, 5])
result = a.gcd(other=b)
a = torch.tensor([5, 10, 15])
b = torch.tensor([3, 4, 5])
a.gcd_(b)
a = torch.tensor([5, 10, 15])
b = torch.tensor([3, 4, 5])
a.gcd_(other=b)
result = torch.tensor([[1, 2], [3, 4]]).ge(torch.tensor([[1, 1], [4, 4]]))
input = torch.tensor([[1, 2], [3, 4]])
other = torch.tensor([[1, 1], [4, 4]])
result = input.ge(other)
result = torch.tensor([-0.6341, -1.4208, -1.09, 0.5826]).geometric_(0.5)
input = torch.tensor([-0.6341, -1.4208, -1.09, 0.5826])
result = input.geometric_(0.5)
x = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]).cuda()
result = x.get_device()
result = None
x = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]).cpu()
result = x.get_device()
src = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
result = src.grad
result = torch.tensor([[1, 2], [3, 4]]).greater(torch.tensor([[1, 1], [4, 4]]))
input = torch.tensor([[1, 2], [3, 4]])
other = torch.tensor([[1, 1], [4, 4]])
result = input.greater(other)
result = torch.tensor([[1, 2], [3, 4]]).greater_equal(torch.tensor([[1, 1],
    [4, 4]]))
input = torch.tensor([[1, 2], [3, 4]])
other = torch.tensor([[1, 1], [4, 4]])
result = input.greater_equal(other)
result = torch.tensor([[1, 2], [3, 4]]).gt(torch.tensor([[1, 1], [4, 4]]))
input = torch.tensor([[1, 2], [3, 4]])
other = torch.tensor([[1, 1], [4, 4]])
result = input.gt(other)
a = torch.tensor([1.0, 2, 3])
b = torch.tensor([4.0, 5, 6])
result = a.hypot_(b)
a = torch.tensor([1.0, 2, 3])
b = torch.tensor([4.0, 5, 6])
result = a.hypot_(other=b)
a = torch.tensor([1.0, 1.2661, 2.2796])
a.i0()
a = torch.tensor([1.0, 1.2661, 2.2796]).i0()
a = torch.tensor([1.0, 1.2661, 2.2796])
a.i0_()
a = torch.tensor([1.0, 1.2661, 2.2796]).i0_()
x = torch.ones([5, 3])
t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
index = torch.tensor([0, 4, 2])
result = x.index_add(0, index, t)
x = torch.ones([5, 3])
t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
index = torch.tensor([0, 4, 2])
result = x.index_add(dim=0, index=index, source=t)
x = torch.ones([5, 3])
t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
index = torch.tensor([0, 4, 2])
x.index_add_(0, index, t)
x = torch.ones([5, 3])
t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
index = torch.tensor([0, 4, 2])
x.index_add_(dim=0, index=index, source=t)
x = torch.eye(2, 4)
indices = torch.tensor([0, 1])
value = -1
result = x.index_fill(0, indices, value)
indices = torch.tensor([0, 1])
value = -1
result = torch.eye(3, 4).index_fill(1, indices, value)
x = torch.eye(2, 4)
indices = torch.tensor([0, 1])
value = -1
result = x.index_fill_(0, indices, value)
x = torch.eye(3, 4)
indices = torch.tensor([0, 1])
value = -1
result = x.index_fill_(1, indices, value)
x = torch.ones([5, 3])
t = torch.tensor([1.0], dtype=torch.float)
indices = [torch.tensor(i) for i in [[0, 0], [0, 1]]]
result = x.index_put(indices, t)
x = torch.ones([5, 3])
t = torch.tensor([1.0], dtype=torch.float)
indices = [torch.tensor(i) for i in [[0, 0], [0, 1]]]
result = x.index_put(indices, values=t)
x = torch.ones([5, 3])
t = torch.tensor([1.0], dtype=torch.float)
indices = [torch.tensor(i) for i in [[0, 0], [0, 1]]]
x.index_put_(indices, t)
x = torch.ones([5, 3])
t = torch.tensor([1.0], dtype=torch.float)
indices = [torch.tensor(i) for i in [[0, 0], [0, 1]]]
x.index_put_(indices, values=t)
x = torch.eye(2, 4)
indices = torch.tensor([0, 1])
result = x.index_select(0, indices)
indices = torch.tensor([0, 1])
result = torch.eye(3, 4).index_select(1, indices)
i = [[0, 1, 1], [2, 0, 2]]
v = [3, 4, 5]
x = paddle.sparse.sparse_coo_tensor(indices=i, values=v, shape=(2, 3)
    ).coalesce()
result = x.indices()
x = torch.tensor([[0.7308, 1.006, 0.527, 1.4516], [-0.1383, 1.5706, 0.4724,
    0.4141], [0.1193, 0.2829, 0.9037, 0.3957], [-0.8202, -0.6474, -0.1631, 
    -0.6543]])
result = x.inverse()
x = torch.tensor([[[[-0.1533, 2.302, -0.1771, 0.5928], [0.4338, -0.6537, 
    0.2296, 0.5946], [-0.4932, 1.8386, -0.1039, 1.044], [0.1735, -0.8303, -
    0.3821, -0.4384]], [[-0.1533, 2.302, -0.1771, 0.5928], [0.4338, -0.6537,
    0.2296, 0.5946], [-0.4932, 1.8386, -0.1039, 1.044], [0.1735, -0.8303, -
    0.3821, -0.4384]]]])
result = x.inverse()
i = torch.tensor([[0, 1, 1], [2, 0, 2]])
v = torch.tensor([3, 4, 5])
x = paddle.sparse.sparse_coo_tensor(indices=i, values=v, shape=[2, 4])
result = x.is_coalesced()
a = torch.tensor([[4, 9], [23, 2]])
result = a.is_complex()
result = torch.tensor([[4, 9], [23, 2]], dtype=torch.complex64).is_complex()
a = torch.tensor([[4, 9], [23, 2]])
result = a.is_contiguous()
result = torch.tensor([[4, 9], [23, 2]], dtype=torch.complex64).is_contiguous()
x = torch.zeros(5, 3).cpu()
result = x.is_cuda
a = torch.tensor([[4, 9], [23, 2]], dtype=torch.int64)
result = a.is_floating_point()
a = torch.tensor([[4, 9], [23, 2]], dtype=torch.float64)
result = a.is_floating_point()
src = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
result = src.is_leaf
result = torch.tensor([10000.0, 1e-07]).isclose(torch.tensor([10000.1, 1e-08]))
result = torch.tensor([10000.0, 1e-08]).isclose(torch.tensor([10000.1, 1e-09]))
result = torch.tensor([1, float('inf'), 2, float('-inf'), float('nan')]
    ).isfinite()
input = torch.tensor([1, float('inf'), 2, float('-inf'), float('nan')])
result = input.isfinite()
result = torch.tensor([1, float('inf'), 2, float('-inf'), float('nan')]).isinf(
    )
input = torch.tensor([1, float('inf'), 2, float('-inf'), float('nan')])
result = input.isinf()
result = torch.tensor([1, float('inf'), 2, float('-inf'), float('nan')]).isnan(
    )
input = torch.tensor([1, float('inf'), 2, float('-inf'), float('nan')])
result = input.isnan()
result = torch.tensor([1, float('inf'), 2, float('-inf'), float('nan')]
    ).isneginf()
input = torch.tensor([1, float('inf'), 2, float('-inf'), float('nan')])
result = input.isneginf()
result = torch.tensor([1, float('inf'), 2, float('-inf'), float('nan')]
    ).isposinf()
input = torch.tensor([1, float('inf'), 2, float('-inf'), float('nan')])
result = input.isposinf()
x = torch.tensor([1, 1 + 1.0j, 2 + 0.0j])
result = x.isreal()
x = torch.tensor([-0.0, -2.1, 2.5])
result = x.isreal()
x = torch.tensor([[[5.975718021392822 + 0.0j, 5.975718021392822 + 0.0j, 
    5.341437339782715 + 0.0j, 5.404394626617432 + 0.0j, 5.404394626617432 +
    0.0j], [0.0629572868347168 + 0.0j, 0.0629572868347168j, -
    0.0629572868347168 - 0.6342806816101074j, 0.6342806816101074 + 0.0j, 
    0.6342806816101074j], [-0.4979677200317383 + 0.0j, 0.4979677200317383 +
    0.0j, 0.13631296157836914 + 0.0j, -0.19927024841308594 + 0.0j, 
    0.19927024841308594 + 0.0j]]])
result = x.istft(n_fft=4)
x = torch.tensor([[[5.975718021392822 + 0.0j, 5.975718021392822 + 0.0j, 
    5.341437339782715 + 0.0j, 5.404394626617432 + 0.0j, 5.404394626617432 +
    0.0j], [0.0629572868347168 + 0.0j, 0.0629572868347168j, -
    0.0629572868347168 - 0.6342806816101074j, 0.6342806816101074 + 0.0j, 
    0.6342806816101074j], [-0.4979677200317383 + 0.0j, 0.4979677200317383 +
    0.0j, 0.13631296157836914 + 0.0j, -0.19927024841308594 + 0.0j, 
    0.19927024841308594 + 0.0j]]])
result = x.istft(n_fft=4, center=False)
a = torch.tensor([4])
result = a.item()
a = torch.tensor([-1])
result = a.itemsize
a = torch.tensor([-1, -2, 3])
result = a.itemsize
a = torch.tensor([5, 10, 15])
b = torch.tensor([3, 4, 5])
result = a.lcm(b)
a = torch.tensor([5, 10, 15])
b = torch.tensor([3, 4, 5])
result = a.lcm(other=b)
a = torch.tensor([5, 10, 15])
b = torch.tensor([3, 4, 5])
a.lcm_(b)
a = torch.tensor([5, 10, 15])
b = torch.tensor([3, 4, 5])
a.lcm_(other=b)
a = torch.tensor([1.0, 2.0, -3.0, -4.0, 5.0])
b = torch.tensor([1.0, 2.0, -3.0, -4.0, 5.0])
a.ldexp_(b)
a = torch.tensor([1.0, 2.0, -3.0, -4.0, 5.0])
a.ldexp_(other=torch.tensor([1.0, 2.0, -3.0, -4.0, 5.0]))
result = torch.tensor([[1, 2], [3, 4]]).le(torch.tensor([[1, 1], [4, 4]]))
input = torch.tensor([[1, 2], [3, 4]])
other = torch.tensor([[1, 1], [4, 4]])
result = input.le(other)
result = torch.tensor([1.0, 2.0, 3.0, 4.0]).lerp(torch.tensor([10.0, 10.0, 
    10.0, 10.0]), 0.5)
result = torch.tensor([1.0, 2.0, 3.0, 4.0]).lerp(end=torch.tensor([10.0, 
    10.0, 10.0, 10.0]), weight=0.5)
start = torch.tensor([1.0, 2.0, 3.0, 4.0])
result = start.lerp_(torch.tensor([10.0, 10.0, 10.0, 10.0]), 0.5)
start = torch.tensor([1.0, 2.0, 3.0, 4.0])
result = start.lerp_(end=torch.tensor([10.0, 10.0, 10.0, 10.0]), weight=0.5)
result = torch.tensor([[1, 2], [3, 4]]).less(torch.tensor([[1, 1], [4, 4]]))
input = torch.tensor([[1, 2], [3, 4]])
other = torch.tensor([[1, 1], [4, 4]])
result = input.less(other)
result = torch.tensor([[1, 2], [3, 4]]).less_equal(torch.tensor([[1, 1], [4,
    4]]))
input = torch.tensor([[1, 2], [3, 4]])
other = torch.tensor([[1, 1], [4, 4]])
result = input.less_equal(other)
input = torch.tensor([0.34, 1.5, 0.73])
result = input.lgamma()
result = torch.tensor([0.34, 1.5, 0.73]).lgamma()
input = torch.tensor([0.34, 1.5, 0.73])
input.lgamma_()
input = torch.tensor([0.34, 1.5, 0.73]).lgamma_()
input = torch.tensor([4.7767, 4.3234, 1.2156, 0.2411, 4.5739])
result = input.log()
result = torch.tensor([4.7767, 4.3234, 1.2156, 0.2411, 4.5739]).log()
input = torch.tensor([4.7767, 4.3234, 1.2156, 0.2411, 4.5739])
result = input.log10()
result = torch.tensor([4.7767, 4.3234, 1.2156, 0.2411, 4.5739]).log10()
input = torch.tensor([4.7767, 4.3234, 1.2156, 0.2411, 4.5739])
input.log10_()
input = torch.tensor([4.7767, 4.3234, 1.2156, 0.2411, 4.5739]).log10_()
input = torch.tensor([4.7767, 4.3234, 1.2156, 0.2411, 4.5739])
result = input.log1p()
result = torch.tensor([4.7767, 4.3234, 1.2156, 0.2411, 4.5739]).log1p()
input = torch.tensor([4.7767, 4.3234, 1.2156, 0.2411, 4.5739])
input.log1p_()
input = torch.tensor([4.7767, 4.3234, 1.2156, 0.2411, 4.5739]).log1p_()
input = torch.tensor([4.7767, 4.3234, 1.2156, 0.2411, 4.5739])
result = input.log2()
result = torch.tensor([4.7767, 4.3234, 1.2156, 0.2411, 4.5739]).log2()
input = torch.tensor([4.7767, 4.3234, 1.2156, 0.2411, 4.5739])
input.log2_()
input = torch.tensor([4.7767, 4.3234, 1.2156, 0.2411, 4.5739]).log2_()
input = torch.tensor([4.7767, 4.3234, 1.2156, 0.2411, 4.5739])
input.log_()
input = torch.tensor([4.7767, 4.3234, 1.2156, 0.2411, 4.5739]).log_()
result = torch.tensor([True, False, True]).logical_and(torch.tensor([True, 
    False, False]))
a = torch.tensor([0, 1, 10, 0], dtype=torch.int8)
b = torch.tensor([4, 0, 1, 0], dtype=torch.int8)
result = a.logical_and(b)
result = torch.tensor([True, False, True]).logical_not()
a = torch.tensor([0, 1, 10, 0], dtype=torch.int8)
result = a.logical_not()
result = torch.tensor([True, False, True]).logical_or(torch.tensor([True, 
    False, False]))
a = torch.tensor([0, 1, 10, 0], dtype=torch.int8)
b = torch.tensor([4, 0, 1, 0], dtype=torch.int8)
result = a.logical_or(b)
result = torch.tensor([True, False, True]).logical_xor(torch.tensor([True, 
    False, False]))
a = torch.tensor([0, 1, 10, 0], dtype=torch.int8)
b = torch.tensor([4, 0, 1, 0], dtype=torch.int8)
result = a.logical_xor(b)
input = torch.tensor([0.2796, 0.9331, 0.6486, 0.1523, 0.6516])
result = input.logit(eps=1e-06)
input = torch.tensor([0.2796, 0.9331, 0.6486, 0.1523, 0.6516])
eps = 1e-06
result = input.logit(eps)
input = torch.tensor([0.2796, 0.9331, 0.6486, 0.1523, 0.6516])
input.logit_(eps=1e-06)
input = torch.tensor([0.2796, 0.9331, 0.6486, 0.1523, 0.6516])
eps = 1e-06
input.logit_(eps)
input = torch.tensor([1.4907, 1.0593, 1.5696])
result = input.logsumexp(0)
input = torch.tensor([[1.4907, 1.0593, 1.5696], [1.4907, 1.0593, 1.5696]])
result = input.logsumexp(1)
result = torch.tensor([[1, 2], [3, 4]]).lt(torch.tensor([[1, 1], [4, 4]]))
input = torch.tensor([[1, 2], [3, 4]])
other = torch.tensor([[1, 1], [4, 4]])
result = input.lt(other)
A = torch.tensor([[[0.3591, -0.0479, -0.2174], [-0.6957, -1.4667, 1.4384],
    [0.0735, 0.1147, 0.0513]], [[-1.2565, -2.1263, 0.8075], [-0.3665, -
    3.354, -0.9417], [-0.1299, -0.0689, -0.6207]]])
A_LU, pivots = A.lu()
A = torch.tensor([[[0.3591, -0.0479, -0.2174], [-0.6957, -1.4667, 1.4384],
    [0.0735, 0.1147, 0.0513]], [[-1.2565, -2.1263, 0.8075], [-0.3665, -
    3.354, -0.9417], [-0.1299, -0.0689, -0.6207]]])
A_LU, pivots, info = A.lu(get_infos=True)
x = torch.tensor([[0.0 + 0.0j, 1.0 + 1.0j], [2.0 + 2.0j, 3.0 + 3.0j]])
result = x.mT
x = torch.tensor([[1, 2, 3], [3, 4, 6]])
result = x.mT
a = torch.Tensor([[1.0, 0.2], [0.3, 0.4]])
b = torch.Tensor([[1, 0], [1, 1]]) == 1
result = a.masked_fill(b, 2)
a = torch.Tensor([[1.0, 0.2], [0.3, 0.4]])
b = torch.Tensor([[1, 0], [1, 1]]) == 1
result = a.masked_fill(mask=b, value=2)
a = torch.Tensor([[1.0, 0.2], [0.3, 0.4]])
b = torch.Tensor([[1, 0], [1, 1]]) == 1
result = a.masked_fill_(b, 2)
a = torch.Tensor([[1.0, 0.2], [0.3, 0.4]])
b = torch.Tensor([[1, 0], [1, 1]]) == 1
result = a.masked_fill_(mask=b, value=2)
x = torch.tensor([0, 0, 0, 0, 0])
mask = torch.tensor([[0, 0, 0, 1, 1], [1, 1, 0, 1, 1]], dtype=torch.bool)
source = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
result = x.masked_scatter(mask, source)
x = torch.tensor([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
mask = torch.tensor([[0, 0, 0, 1, 1], [1, 1, 0, 1, 1]], dtype=torch.bool)
source = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
result = x.masked_scatter(mask, source)
x = torch.tensor([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
mask = torch.tensor([[0, 0, 0, 1, 1], [1, 1, 0, 1, 1]], dtype=torch.bool)
source = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
result = x.masked_scatter_(mask, source)
x = torch.eye(2, 4)
mask = x > 0
result = x.masked_select(mask)
x = torch.ones(2, 4)
result = x.masked_select(x > 0)
x = torch.tensor([[4.0, 5.0, 6.0], [1.0, 2.0, 3.0], [4.0, 9.0, 10.0]])
y = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
result = x.matmul(y)
x = torch.tensor([[4.0, 5.0, 6.0], [1.0, 2.0, 3.0], [4.0, 9.0, 10.0]])
y = torch.tensor([1.0, 2.0, 3.0])
result = x.matmul(y)
x = torch.tensor([[4.0, 5.0, 6.0], [1.0, 2.0, 3.0], [4.0, 9.0, 10.0]])
result = x.matrix_power(2)
x = torch.tensor([[4.0, 5.0, 6.0], [1.0, 2.0, 3.0], [4.0, 9.0, 10.0]])
result = x.matrix_power(-2)
result = torch.tensor([[1, 2], [3, 4]]).maximum(torch.tensor([[1, 1], [4, 4]]))
input = torch.tensor([[1, 2], [3, 4]])
other = torch.tensor([[1, 1], [4, 4]])
result = input.maximum(other)
input = torch.tensor([1.4907, 1.0593, 1.5696])
result = input.mean()
input = torch.tensor([[1.4907, 1.0593, 1.5696], [1.4907, 1.0593, 1.5696]])
result = input.mean(1)
result = torch.tensor([[1, 2], [3, 4]]).minimum(torch.tensor([[1, 1], [4, 4]]))
input = torch.tensor([[1, 2], [3, 4]])
other = torch.tensor([[1, 1], [4, 4]])
result = input.minimum(other)
a = torch.tensor([[1.0, 2.0], [4.0, 5.0]])
b = torch.tensor([[1.0, 3.0], [3.0, 6.0]])
result = a.mm(b)
a = torch.tensor([[1.0, 2.0], [4.0, 5.0]])
b = torch.tensor([[1.0, 3.0], [3.0, 6.0]])
result = a.mm(mat2=b)
x = torch.arange(24)
x = torch.reshape(x, (1, 4, 6))
result = x.moveaxis(1, 0)
x = torch.arange(24)
x = torch.reshape(x, (1, 4, 6))
result = x.moveaxis((1, 0), (0, 1))
x = torch.tensor([[-1.3029, 0.4921, -0.7432], [2.6672, -0.0987, 0.075], [
    0.1436, -1.0114, 1.3641]])
result = x.msort()
x = torch.tensor([[-1.3029, 0.4921, -0.7432], [2.6672, -0.0987, 0.075], [
    0.1436, -1.0114, 1.3641]])
result = x.msort() * 3.0
input = torch.tensor([0.2015, -0.4255, 2.6087])
other = torch.tensor([0.2015, -0.4255, 2.6087])
result = input.mul(other)
input = torch.tensor([0.2015, -0.4255, 2.6087])
other = torch.tensor([2, 6, 4])
result = input.mul(other=other)
input = torch.tensor([0.2015, -0.4255, 2.6087])
other = torch.tensor([0.2015, -0.4255, 2.6087])
result = input.mul_(other)
input = torch.tensor([0.2015, -0.4255, 2.6087])
result = input.mul_(other=5.0)
torch.manual_seed(100)
weights = torch.tensor([0, 10, 3, 0], dtype=torch.float)
result = weights.multinomial(2)
torch.manual_seed(100)
weights = torch.tensor([0, 10, 3, 0], dtype=torch.float)
result = weights.multinomial(4, replacement=True)
input = torch.tensor([0.2015, -0.4255, 2.6087])
other = torch.tensor([0.2015, -0.4255, 2.6087])
result = input.multiply(other)
input = torch.tensor([0.2015, -0.4255, 2.6087])
other = torch.tensor([2, 6, 4])
result = input.multiply(other=other)
input = torch.tensor([0.2015, -0.4255, 2.6087])
other = torch.tensor([0.2015, -0.4255, 2.6087])
result = input.multiply_(other)
input = torch.tensor([0.2015, -0.4255, 2.6087])
result = input.multiply_(other=5.0)
a = torch.tensor([[1.0, 2.0], [4.0, 5.0]])
b = torch.tensor([1.0, 3.0])
result = a.mv(b)
a = torch.tensor([[1.0, 2.0], [4.0, 5.0]])
b = torch.tensor([1.0, 3.0])
result = a.mv(vec=b)
input = torch.tensor([[1, 2], [3.0, float('nan')]])
result = input.nan_to_num()
input = torch.tensor([float('nan'), float('inf'), -float('inf'), 3.14])
result = input.nan_to_num(0.0, 1.0, -1.0)
input = torch.tensor([[1, 2], [3.0, float('nan')]])
input.nan_to_num_()
input = torch.tensor([float('nan'), float('inf'), -float('inf'), 3.14])
input.nan_to_num_(0.0, 1.0, -1.0)
x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
result = x.narrow(0, 0, 2)
x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
result = x.narrow(1, 1, 2)
result = torch.tensor([1, 0.5]).ndim
a = torch.tensor([[1, 0.5]])
result = a.ndim
result = torch.tensor([1, 0.5]).ndimension()
a = torch.tensor([[1, 0.5]])
result = a.ndimension()
result = torch.tensor([[1, 2], [3, 4]]).ne(torch.tensor([[1, 1], [4, 4]]))
input = torch.tensor([[1, 2], [3, 4]])
other = torch.tensor([[1, 1], [4, 4]])
result = input.ne(other)
result = torch.tensor([-1, -2, 3]).neg()
a = torch.tensor([-1, -2, 3])
result = a.neg()
a = torch.tensor([-1, -2, 3]).neg_()
a = torch.tensor([-1, -2, 3])
a.neg_()
x = torch.tensor([1.0, 2.0, 3.0])
result = x.new_empty((1,))
x = torch.tensor([1.0, 2.0, 3.0])
result = x.new_empty((1, 3), dtype=torch.float64)
x = torch.tensor([1.0, 2.0, 3.0])
result = x.new_full((1,), 3.1234)
x = torch.tensor([1.0, 2.0, 3.0])
result = x.new_full((1, 3), 3.1234, dtype=torch.float64)
x = torch.tensor([1.0, 2.0, 3.0])
result = x.new_ones((1,))
x = torch.tensor([1.0, 2.0, 3.0])
result = x.new_ones((1, 3), dtype=torch.float64)
x = torch.tensor([1.0, 2.0, 3.0])
result = x.new_zeros((1,))
x = torch.tensor([1.0, 2.0, 3.0])
result = x.new_zeros((1, 3), dtype=torch.float64)
input = torch.tensor([1.0, 2.0])
result = input.nextafter(torch.tensor([2.0, 1.0]))
input = torch.tensor([1.0, 2.0])
b = torch.tensor([2.0, 1.0])
result = input.nextafter(b)
result = torch.tensor([1, 1, 1, 0, 1]).nonzero()
result = torch.tensor([[0.6, 0.0, 0.0, 0.0], [0.0, 0.4, 0.0, 0.0], [0.0, 
    0.0, 1.2, 0.0], [0.0, 0.0, 0.0, -0.4]]).nonzero()
input = torch.tensor([[[-12.0, -11.0, -10.0, -9.0], [-8.0, -7.0, -6.0, -5.0
    ], [-4.0, -3.0, -2.0, -1.0]], [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 
    7.0], [8.0, 9.0, 10.0, 11.0]]])
result = input.norm(p='fro')
input = torch.tensor([[-12.0, -11.0, -10.0, -9.0], [-8.0, -7.0, -6.0, -5.0],
    [-4.0, -3.0, -2.0, -1.0]])
result = input.norm(p='nuc')
result = torch.Tensor([[1.0, 2.0], [3.0, 4.0]])
result.normal_()
result = torch.Tensor([[1.0, 2.0], [3.0, 4.0]])
result.normal_(0, 1)
result = torch.tensor([[1, 2], [3, 4]]).not_equal(torch.tensor([[1, 1], [4,
    4]]))
input = torch.tensor([[1, 2], [3, 4]])
other = torch.tensor([[1, 1], [4, 4]])
result = input.not_equal(other)
x = torch.tensor([1.0, 2, 3])
y = torch.tensor([1.0, 2, 3, 4])
result = x.outer(y)
x = torch.tensor([1.0, 2, 3])
y = torch.tensor([1.0, 2, 3, 4])
result = x.outer(vec2=y)
x = torch.tensor([1.0, 2.0, 3.0, 4.0])
result = x.permute(0)
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
result = x.permute(0, 1)
a = torch.Tensor([[1.0, 2.0], [3.0, 4.0]])
result = a.pin_memory()
x = torch.tensor([1.02, 2.21, 3.33])
result = x.polygamma(1)
x = torch.tensor([1.02, 2.21, 3.33, 4])
result = x.polygamma(1)
x = torch.tensor([1.02, 2.21, 3.33])
x.polygamma_(1)
x = torch.tensor([1.02, 2.21, 3.33, 4])
x.polygamma_(n=1)
a = torch.tensor([0.4331, 1.2475, 0.6834, -0.2791])
result = a.pow(2)
a = torch.tensor([0.4331, 1.2475, 0.6834, -0.2791])
b = torch.tensor([1, 2, 3, 4])
result = a.pow(b)
input = torch.tensor([1.4907, 1.0593, 1.5696])
result = input.prod()
input = torch.tensor([[1.4907, 1.0593, 1.5696], [1.4907, 1.0593, 1.5696]])
result = input.prod(1)
x = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
result = x.quantile(0.6)
x = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
result = x.quantile(q=0.6)
x = torch.tensor([[3.142, -3.142], [6.283, -6.283], [1.57, -1.57]])
result = x.rad2deg()
result = torch.tensor([[3.142, -3.142], [6.283, -6.283], [1.57, -1.57]]
    ).rad2deg()
result = torch.tensor([-0.6341, -1.4208, -1.09, 0.5826]).random_(0, 5)
input = torch.tensor([-0.6341, -1.4208, -1.09, 0.5826])
result = input.random_(0, 5)
a = torch.tensor([[4, 9], [23, 2]])
result = a.ravel()
result = torch.tensor([[4, 9], [23, 2]]).ravel()
result = torch.tensor([-0.4595, -2.1219, -1.4314, 0.7298]).reciprocal()
a = torch.tensor([-0.4595, -2.1219, -1.4314, 0.7298])
result = a.reciprocal()
result = torch.tensor([-0.4595, -2.1219, -1.4314, 0.7298]).reciprocal_()
a = torch.tensor([-0.4595, -2.1219, -1.4314, 0.7298])
result = a.reciprocal_()
v = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
h = v.register_hook(lambda grad: grad * 2)
v.backward(grad_tensor=torch.tensor([1.0, 2.0, 3.0]))
result = torch.tensor(v.grad)
v = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
h = v.register_hook(hook=lambda grad: grad * 2)
v.backward(grad_tensor=torch.tensor([1.0, 2.0, 3.0]))
result = torch.tensor(v.grad)
a = torch.tensor([-3.0, -2, -1, 1, 2, 3])
result = a.remainder(torch.tensor(2.0))
a = torch.tensor([1, 2, 3, 4, 5])
result = a.remainder(torch.tensor(1.5))
x = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
result = x.renorm(1, 0, 5)
x = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
result = x.renorm(p=1, dim=0, maxnorm=5)
x = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
x.renorm_(1, 0, 5)
x = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
x.renorm_(p=1, dim=0, maxnorm=5)
x = torch.tensor([1, 2, 3])
result = x.repeat(4)
x = torch.tensor([1, 2, 3])
result = x.repeat(4, 2, 3)
a = torch.tensor([[4, 9], [23, 2]])
result = a.repeat_interleave(3, 0)
a = torch.tensor([[4, 9], [23, 2]])
result = a.repeat_interleave(repeats=3, dim=1)
data = torch.tensor([23.0, 32.0, 43.0])
result = 1
>>>>>>if not torch.utils.data.requires_grad:
    result = 2
data = torch.tensor([23.0, 32.0, 43.0])
>>>>>>result = torch.utils.data.requires_grad
a = torch.Tensor([[1.0, 2.0], [3.0, 4.0]])
result = a.requires_grad_()
a = torch.Tensor([[1.0, 2.0], [3.0, 4.0]])
result = a.requires_grad_(True)
a = torch.arange(4.0)
result = a.reshape(2, 2)
a = torch.arange(4.0)
result = a.reshape((2, 2))
x = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
result = x.roll(1)
x = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
result = x.roll(1, 0)
a = torch.tensor([[0.9254, -0.6213]])
result = a.round()
a = torch.tensor([[102003.9254, -12021.6213]])
result = a.round(decimals=1)
a = torch.tensor([[0.9254, -0.6213]])
result = a.round_()
a = torch.tensor([[102003.9254, -12021.6213]])
result = a.round_(decimals=1)
result = torch.tensor([0.297, 1.542, 4]).rsqrt()
a = torch.tensor([0.297, 1.542, 4])
result = a.rsqrt()
result = torch.tensor([0.297, 1.542, 4]).rsqrt_()
result = torch.tensor([0.297, 1.542, 4])
result.rsqrt_()
x = torch.arange(15).reshape([3, 5]).astype(torch.float32)
index = torch.tensor([[0], [1], [2]])
result = x.scatter(1, index, 1.0)
x = torch.arange(15).reshape([3, 5]).astype(torch.float32)
index = torch.tensor([[0], [1], [2]])
result = x.scatter(dim=1, index=index, value=1.0)
index = torch.tensor([[0], [1], [2]])
result = torch.zeros(3, 5).scatter_(1, index, 1.0)
index = torch.tensor([[0], [1], [2]])
result = torch.zeros(3, 5).scatter_(dim=1, index=index, value=1.0)
src = torch.ones((1, 5))
index = torch.tensor([[0, 1, 2, 0, 0]])
result = torch.zeros(3, 5, dtype=src.dtype).scatter_add(0, index, src)
src = torch.ones((2, 5))
index = torch.tensor([[0, 1, 2, 0, 0], [0, 1, 2, 2, 2]])
result = torch.zeros(3, 5, dtype=src.dtype).scatter_add(dim=0, index=index,
    src=src)
src = torch.ones((1, 5))
index = torch.tensor([[0, 1, 2, 0, 0]])
result = torch.zeros(3, 5, dtype=src.dtype).scatter_add_(0, index, src)
src = torch.ones((2, 5))
index = torch.tensor([[0, 1, 2, 0, 0], [0, 1, 2, 2, 2]])
result = torch.zeros(3, 5, dtype=src.dtype).scatter_add_(0, index, src)
src = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
index = torch.tensor([0, 1, 0, 1, 2, 1])
input = torch.tensor([1.0, 2.0, 3.0, 4.0])
type = 'sum'
result = input.scatter_reduce(0, index, src, reduce=type)
src = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
index = torch.tensor([0, 1, 0, 1, 2, 1])
input = torch.tensor([1.0, 2.0, 3.0, 4.0])
re_type = 'sum'
result = input.scatter_reduce(dim=0, index=index, src=src, reduce=re_type,
    include_self=False)
a = torch.tensor([0.595, -0.0872, 2.3298, -0.2972])
result = a.sgn()
a = torch.tensor([0.595 + 0.3451j, -0.0872 - 0.3451j, 2.3298 + 0.3451j, -
    0.2972 + 0.3451j])
result = a.sgn()
src = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
result = src.shape
src = torch.empty((2, 0))
result = src.shape
a = torch.Tensor([[1.0, 2.0], [3.0, 4.0]])
result = a.sigmoid()
a = torch.Tensor([[1.0, 2.0], [3.0, 4.0]])
a.sigmoid_()
result = torch.tensor([0.9213, 1.0887, -0.8858, -1.7683]).sign()
a = torch.tensor([0.9213, 1.0887, -0.8858, -1.7683])
result = a.sign()
x = torch.tensor([-0.0, 1.1, -2.1, 0.0, 2.5], dtype=torch.float32)
result = x.signbit()
x = torch.tensor([-0.0, 1.1, -2.1, 0.0, 2.5], dtype=torch.float64)
result = x.signbit()
a = torch.Tensor([[1.0, 2.0], [3.0, 4.0]])
result = a.sin()
a = torch.tensor([1.4309, 1.2706, -0.8562, 0.9796])
a.sin_()
a = torch.tensor([0.595, -0.0872, 0, -0.2972])
result = a.sinc()
result = torch.tensor([0.595, -0.0872, 0, -0.2972]).sinc()
a = torch.tensor([0.595, -0.0872, 0, -0.2972])
result = a.sinc_()
result = torch.tensor([0.595, -0.0872, 0, -0.2972]).sinc_()
a = torch.Tensor([[1.0, 2.0], [3.0, 4.0]])
result = a.sinh()
a = torch.tensor([1.4309, 1.2706, -0.8562, 0.9796])
a.sinh_()
a = torch.tensor([0.595, -0.0872, 0, -0.2972])
result = a.size()
a = torch.tensor([0.595, -0.0872, 0, -0.2972])
result = a.size(dim=0)
input = torch.tensor([[-1.2837, -0.0297, 0.0355], [0.9112, -1.7526, -0.4061]])
result = input.softmax(dim=0)
input = torch.tensor([[-1.2837, -0.0297, 0.0355], [0.9112, -1.7526, -0.4061]])
result = input.softmax(1)
i = torch.tensor([[0, 1, 1], [2, 0, 2]])
v = torch.tensor([3, 4, 5], dtype=torch.float32)
x = paddle.sparse.sparse_coo_tensor(indices=i, values=v, shape=[2, 4])
result = x.sparse_dim()
result = torch.tensor([0.297, 1.542, 4]).sqrt()
a = torch.tensor([0.297, 1.542, 4])
result = a.sqrt()
result = torch.tensor([0.297, 1.542, 4]).sqrt_()
result = torch.tensor([0.297, 1.542, 4])
result.sqrt_()
result = torch.tensor([0.297, 1.542, 4]).square()
a = torch.tensor([0.297, 1.542, 4])
result = a.square()
x = torch.zeros(2, 1, 2, 1, 2)
result = x.squeeze()
result = torch.zeros(2, 1, 2, 1, 2).squeeze()
input = torch.tensor([1.4907, 1.0593, 1.5696])
result = input.std(unbiased=False)
input = torch.tensor([[1.4907, 1.0593, 1.5696], [1.4907, 1.0593, 1.5696]])
result = input.std(unbiased=False)
a = torch.tensor([0.9254, -0.6213])
result = a.stride(dim=0)
a = torch.tensor([[0.9254, -0.6213], [0.9254, -0.6213]])
result = a.stride(dim=None)
x = torch.tensor([1, 2, 3])
result = x.sub(torch.tensor([1, 4, 6]))
x = torch.tensor([1, 2, 3])
result = x.sub(20)
a = torch.tensor([0.595, -0.0872, 2.3298, -0.2972])
b = torch.tensor([1.0, 2.0, 3.0, 4.0])
a.sub_(b)
a = torch.tensor([0.595, -0.0872, 2.3298, -0.2972])
b = torch.tensor([1.0, 2.0, 3.0, 4.0])
a.sub_(other=b)
input = torch.tensor([1.4907, 1.0593, 1.5696])
result = input.sum()
input = torch.tensor([[1.4907, 1.0593, 1.5696], [1.4907, 1.0593, 1.5696]])
result = input.sum(1)
x = torch.tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])
result = x.swapaxes(0, 1)
x = torch.tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])
result = x.swapaxes(axis0=0, axis1=1)
x = torch.tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])
result = x.swapdims(0, 1)
x = torch.tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])
result = x.swapdims(dim0=0, dim1=1)
a = torch.Tensor([[1.0, 2.0], [3.0, 4.0]])
result = a.t()
a = torch.Tensor([[1.0, 2.0], [3.0, 4.0]])
result = a.t_()
x = torch.tensor([[1, 2, 3], [3, 4, 6]])
idx = torch.tensor([[0]])
result = x.take_along_dim(idx, 1)
x = torch.tensor([[1, 2, 3], [3, 4, 6]])
idx = torch.tensor([[0]])
result = x.take_along_dim(indices=idx, dim=0)
a = torch.Tensor([[1.0, 2.0], [3.0, 4.0]])
result = a.tan()
a = torch.Tensor([[1.0, 2.0], [3.0, 4.0]])
a.tan_()
a = torch.Tensor([[1.0, 2.0], [3.0, 4.0]])
result = a.tanh()
result = torch.Tensor([[1.0, 2.0], [3.0, 4.0]])
result.tanh_()
a = torch.arange(8)
result = a.tensor_split(4)
a = torch.arange(7)
result = a.tensor_split(sections=3)
cpu = torch.device('cpu')
a = torch.ones(2, 3)
c = torch.ones(2, 3, dtype=torch.float64, device=cpu)
result = a.to(cpu, non_blocking=False, copy=False)
cpu = torch.device('cpu')
a = torch.ones(2, 3)
result = a.to('cpu')
i = torch.tensor([[0, 1, 1], [2, 0, 2]])
v = torch.tensor([3, 4, 5], dtype=torch.float32)
x = paddle.sparse.sparse_coo_tensor(indices=i, values=v, shape=[2, 4])
result = x.to_dense()
a = torch.Tensor([[1.0, 2.0], [3.0, 4.0]])
result = torch.tensor(a.tolist())
a = torch.Tensor([[1.0, 2.0], [3.0, 4.0]])
result = a.topk(2)
a = torch.Tensor([[1.0, 2.0], [3.0, 4.0]])
result = a.topk(2, dim=0)
a = torch.Tensor([[1.0, 2.0], [3.0, 4.0]])
result = a.trace()
a = torch.Tensor([[1.0, 2.0], [3.0, 4.0]])
result = a.transpose(dim0=0, dim1=1)
a = torch.Tensor([[1.0, 2.0], [3.0, 4.0]])
result = a.transpose(0, 1)
a = torch.tensor([[1.3192, 1.9915, 1.9674, 1.7151]])
result = a.tril()
a = torch.tensor([[1.3192, 1.9915, 1.9674, 1.7151]])
result = a.tril(1)
x = torch.tensor([[-1.0813, -0.8619, 0.7105], [0.0935, 0.138, 2.2112], [-
    0.3409, -0.9828, 0.0289]])
x.tril_()
x = torch.tensor([[-1.0813, -0.8619, 0.7105], [0.0935, 0.138, 2.2112], [-
    0.3409, -0.9828, 0.0289]])
x.tril_(1)
a = torch.tensor([[1.3192, 1.9915, 1.9674, 1.7151]])
result = a.triu()
a = torch.tensor([[1.3192, 1.9915, 1.9674, 1.7151]])
result = a.triu(1)
x = torch.tensor([[-1.0813, -0.8619, 0.7105], [0.0935, 0.138, 2.2112], [-
    0.3409, -0.9828, 0.0289]])
x.triu_()
x = torch.tensor([[-1.0813, -0.8619, 0.7105], [0.0935, 0.138, 2.2112], [-
    0.3409, -0.9828, 0.0289]])
x.triu_(1)
a = torch.tensor([4.67, 9.76, 8.53])
b = torch.tensor([3.5, 3.9, 1.83])
result = a.true_divide(b)
a = torch.tensor([[4.0, 9.0, 8.0]])
b = torch.tensor([2.0, 3.0, 4.0])
result = a.true_divide(other=b)
a = torch.Tensor([[1.0, 2.0], [3.0, 4.0]])
result = a.trunc()
a = torch.Tensor([[1.0, 2.0], [3.0, 4.0]])
a.trunc_()
src = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
a = torch.tensor([1])
result = src.type_as(a)
src = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
a = torch.tensor([1])
result = src.type_as(other=a)
a = torch.Tensor([[1.0, 2.0], [3.0, 4.0]])
result = a.unbind()
a = torch.Tensor([[1.0, 2.0], [3.0, 4.0]])
result = a.unbind(dim=0)
a = torch.tensor([[0.218, 1.0558, 0.1608, 0.9245], [1.3794, 1.409, 0.2514, 
    -0.8818], [-0.4561, 0.5123, 1.7505, -0.4094]])
result = a.unflatten(-1, (2, 2))
a = torch.tensor([[0.218, 1.0558, 0.1608, 0.9245], [1.3794, 1.409, 0.2514, 
    -0.8818], [-0.4561, 0.5123, 1.7505, -0.4094]])
result = a.unflatten(1, (2, 2))
x = torch.arange(1.0, 8)
results = x.unfold(0, 2, 1)
x = torch.arange(1.0, 8)
results = x.unfold(0, 2, 2)
result = torch.Tensor([[1.0, 2.0], [3.0, 4.0]])
result.uniform_()
result = torch.Tensor([[1.0, 2.0], [3.0, 4.0]])
result.uniform_(0, to=1)
x = torch.tensor([1, 1, 2, 2, 3, 1, 1, 2])
result = x.unique_consecutive()
x = torch.tensor([1, 1, 2, 2, 3, 1, 1, 2])
result, inverse_indices = x.unique_consecutive(return_inverse=True)
x = torch.zeros(2, 2, 2)
result = x.unsqueeze(0)
result = torch.zeros(2, 2, 1, 2).unsqueeze(3)
i = [[0, 1, 1], [2, 0, 2]]
v = [3, 4, 5]
x = paddle.sparse.sparse_coo_tensor(indices=i, values=v, shape=(2, 3)
    ).coalesce()
result = x.values()
input = torch.tensor([1.4907, 1.0593, 1.5696])
result = input.var(unbiased=False)
input = torch.tensor([[1.4907, 1.0593, 1.5696], [1.4907, 1.0593, 1.5696]])
result = input.var(unbiased=False)
a = torch.arange(4.0)
result = a.view(2, 2)
a = torch.arange(4.0)
result = a.view((2, 2))
a = torch.ones([15])
b = torch.zeros([3, 5])
result = a.view_as(b)
a = torch.ones([15])
b = torch.zeros([3, 5])
result = a.view_as(other=b)
result = torch.Tensor([[1.0, 2.0], [3.0, 4.0]])
result.zero_()
linear = paddle.compat.nn.Linear(5, 5)
result = linear.weight.data.zero_()
result = torch.__version__
result = torch.__version__.split()
result = torch.__version__.split(sep='234')
result = torch.abs(torch.tensor([-1, -2, 3]))
a = torch.tensor([-1, -2, 3])
result = torch.abs(a)
a = torch.tensor([-1])
torch.abs_(a)
a = torch.tensor([-1, -2, 3])
torch.abs_(a)
result = torch.acos(torch.tensor([0.34, -0.56, 0.73]))
a = torch.tensor([0.34, -0.56, 0.73])
result = torch.acos(a)
result = torch.acosh(torch.tensor([1.3192, 1.9915, 1.9674, 1.7151]))
a = torch.tensor([1.3192, 1.9915, 1.9674, 1.7151])
result = torch.acosh(a)
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
result = torch.adaptive_avg_pool1d(x, 5)
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
result = torch.adaptive_avg_pool1d(input=x, output_size=5)
result = torch.add(torch.tensor([1, 2, 3]), torch.tensor([1, 4, 6]))
result = torch.add(torch.tensor([1, 2, 3]), 20)
x = torch.tensor([[1, 2], [4, 5]])
mat1 = torch.tensor([[1, 2], [4, 5]])
mat2 = torch.tensor([[1, 2], [4, 5]])
result = torch.addmm(x, mat1, mat2)
x = torch.tensor([[1.0, 2], [4, 5]])
mat1 = torch.tensor([[1.0, 2], [4, 5]])
mat2 = torch.tensor([[1.0, 2], [4, 5]])
result = torch.addmm(x, mat1, mat2, beta=0.6, alpha=0.7)
a = torch.rand(1, 2).bool()
result = torch.all(a)
a = torch.rand(3, 4)
result = torch.all(a)
x = torch.tensor([[1, 2, 3], [3, 4, 6]])
result = torch.amax(x)
x = torch.tensor([[1, 2, 3], [3, 4, 6]])
result = torch.amax(x, dim=1)
x = torch.tensor([[1, 2, 3], [3, 4, 6]])
result = torch.amin(x)
x = torch.tensor([[1, 2, 3], [3, 4, 6]])
result = torch.amin(x, dim=1)
model = paddle.compat.nn.Linear(10, 5, device='cuda')
input = torch.randn(4, 10, device='cuda')
with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=
    False, cache_enabled=True):
    result = model(input)
model = paddle.compat.nn.Linear(10, 5, device='cuda')
input = torch.randn(4, 10, device='cuda')
with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=
    False, cache_enabled=True):
    result = model(input)
result = torch.angle(torch.tensor([-1 + 1.0j, -2 + 2.0j, 3 - 3.0j]))
x = torch.tensor([-1 + 1.0j, -2 + 2.0j, 3 - 3.0j])
result = torch.angle(x) * 180 / 3.14159
a = torch.rand(1, 2).bool()
result = torch.any(a)
a = torch.rand(3, 4)
result = torch.any(a)
result = torch.arange(5)
result = torch.arange(5.0)
input = torch.tensor([[1.3398, 0.2663, -0.2686, 0.245], [-0.7401, -0.8805, 
    -0.3402, -1.1936], [0.4907, -1.3948, -1.0691, -0.3132], [-1.6092, 
    0.5419, -0.2993, 0.3195]])
result = torch.argmax(input)
input = torch.tensor([[1.3398, 0.2663, -0.2686, 0.245], [-0.7401, -0.8805, 
    -0.3402, -1.1936], [0.4907, -1.3948, -1.0691, -0.3132], [-1.6092, 
    0.5419, -0.2993, 0.3195]])
result = torch.argmax(input, dim=1)
input = torch.tensor([[1.3398, 0.2663, -0.2686, 0.245], [-0.7401, -0.8805, 
    -0.3402, -1.1936], [0.4907, -1.3948, -1.0691, -0.3132], [-1.6092, 
    0.5419, -0.2993, 0.3195]])
result = torch.argmin(input)
input = torch.tensor([[1.3398, 0.2663, -0.2686, 0.245], [-0.7401, -0.8805, 
    -0.3402, -1.1936], [0.4907, -1.3948, -1.0691, -0.3132], [-1.6092, 
    0.5419, -0.2993, 0.3195]])
result = torch.argmin(input, dim=1)
input = torch.tensor([[1.3398, 0.2663, -0.2686, 0.245], [-0.7401, -0.8805, 
    -0.3402, -1.1936], [0.4907, -1.3948, -1.0691, -0.3132], [-1.6092, 
    0.5419, -0.2993, 0.3195]])
result = torch.argsort(input)
input = torch.tensor([[1.3398, 0.2663, -0.2686, 0.245], [-0.7401, -0.8805, 
    -0.3402, -1.1936], [0.4907, -1.3948, -1.0691, -0.3132], [-1.6092, 
    0.5419, -0.2993, 0.3195]])
result = torch.argsort(input, dim=1)
input = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
result = torch.argwhere(input)
input = torch.tensor([0.0, 1.0, 0.0, 3.0])
result = torch.argwhere(input=input)
x = torch.tensor([[0.0335, 0.183, -0.1269], [0.1897, -0.1422, -0.494], [-
    0.7674, -0.0134, -0.3733]])
result = torch.as_strided(x, (2, 2), (1, 2))
x = torch.tensor([[0.0335, 0.183, -0.1269], [0.1897, -0.1422, -0.494], [-
    0.7674, -0.0134, -0.3733]])
result = torch.as_strided(x, (2, 2), (1, 2), 0)
a = numpy.array([1, 2, 3])
result = torch.as_tensor(a)
result = torch.as_tensor(numpy.array([1, 2, 3]))
data = [[0, 1], [2, 3]]
result = torch.asarray(data)
data = [[0, 1], [2, 3]]
result = torch.asarray(data, dtype=torch.float64)
result = torch.asin(torch.tensor([0.34, -0.56, 0.73]))
a = torch.tensor([0.34, -0.56, 0.73])
result = torch.asin(a)
result = torch.asinh(torch.tensor([0.1606, -1.4267, -1.0899, -1.025]))
a = torch.tensor([0.1606, -1.4267, -1.0899, -1.025])
result = torch.asinh(a)
result = torch.atan(torch.tensor([0.2341, 0.2539, -0.6256, -0.6448]))
a = torch.tensor([0.2341, 0.2539, -0.6256, -0.6448])
result = torch.atan(a)
input = torch.tensor([0.9041, 0.0196, -0.3108, -2.4423])
other = torch.tensor([0.2341, 0.2539, -0.6256, -0.6448])
result = torch.atan2(input, other)
result = torch.atan2(torch.tensor([0.9041, 0.0196, -0.3108, -2.4423]),
    torch.tensor([0.2341, 0.2539, -0.6256, -0.6448]))
result = torch.atanh(torch.tensor([-0.9385, 0.2968, -0.8591, -0.1871]))
a = torch.tensor([-0.9385, 0.2968, -0.8591, -0.1871])
result = torch.atanh(a)
result = torch.atleast_1d(torch.tensor(123, dtype=torch.int32))
y = torch.tensor([-1, -2, 3])
result = torch.atleast_1d((torch.tensor(123, dtype=torch.int32), y))
result = torch.atleast_2d(torch.tensor(123, dtype=torch.int32))
y = torch.tensor([-1, -2, 3])
result = torch.atleast_2d((torch.tensor(123, dtype=torch.int32), y))
result = torch.atleast_3d(torch.tensor(123, dtype=torch.int32))
y = torch.tensor([-1, -2, 3])
result = torch.atleast_3d((torch.tensor(123, dtype=torch.int32), y))
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
with torch.autocast(device_type='cpu', enabled=False):
    result = x * x
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
with torch.autocast(device_type='cpu'):
    result = x * x


class cus_tanh(Function):

    @staticmethod
    def forward(ctx, x, func=torch.square):
        ctx.func = func
        y = func(x)
        return y

    @staticmethod
    def backward(ctx, dy):
        grad = dy + 1
        return grad


result = cus_tanh()
a = torch.tensor([[[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]]])
b = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
input = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
result = torch.baddbmm(input, a, b)
a = torch.tensor([[[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]]])
b = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
input = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
result = torch.baddbmm(input, a, b, beta=3)
src = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
result = src.to(torch.bfloat16).to(torch.float)
result = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).to(torch.bfloat16).to(
    torch.float)
input = torch.tensor([4, 3, 6, 3, 4])
weights = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
result = torch.bincount(input)
input = torch.tensor([4, 3, 6, 3, 4])
weights = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
result = torch.bincount(input, weights)
result = torch.bitwise_and(torch.tensor([-1, -2, 3], dtype=torch.int8),
    torch.tensor([1, 0, 3], dtype=torch.int8))
input = torch.tensor([-1, -2, 3], dtype=torch.int8)
other = torch.tensor([1, 0, 3], dtype=torch.int8)
result = torch.bitwise_and(input, other)
input = torch.tensor([-2, -7, 31], dtype=torch.int8)
other = torch.tensor([1, 0, 3], dtype=torch.int8)
result = torch.bitwise_left_shift(input, other)
input = torch.tensor([-2, -7, 31], dtype=torch.int8)
other = torch.tensor([1, 0, 3], dtype=torch.int8)
result = torch.bitwise_left_shift(input=input, other=other)
result = torch.bitwise_not(torch.tensor([-1, -2, 3], dtype=torch.int8))
input = torch.tensor([-1, -2, 3], dtype=torch.int8)
result = torch.bitwise_not(input)
result = torch.bitwise_or(torch.tensor([-1, -2, 3], dtype=torch.int8),
    torch.tensor([1, 0, 3], dtype=torch.int8))
input = torch.tensor([-1, -2, 3], dtype=torch.int8)
other = torch.tensor([1, 0, 3], dtype=torch.int8)
result = torch.bitwise_or(input, other)
input = torch.tensor([-2, -7, 31], dtype=torch.int8)
other = torch.tensor([1, 0, 3], dtype=torch.int8)
result = torch.bitwise_right_shift(input, other)
input = torch.tensor([-2, -7, 31], dtype=torch.int8)
other = torch.tensor([1, 0, 3], dtype=torch.int8)
result = torch.bitwise_right_shift(input=input, other=other)
result = torch.bitwise_xor(torch.tensor([-1, -2, 3], dtype=torch.int8),
    torch.tensor([1, 0, 3], dtype=torch.int8))
input = torch.tensor([-1, -2, 3], dtype=torch.int8)
other = torch.tensor([1, 0, 3], dtype=torch.int8)
result = torch.bitwise_xor(input, other)
result = torch.blackman_window(10)
result = torch.blackman_window(10, dtype=torch.float64)
A = torch.tensor([[0, 1], [1, 0]])
B = torch.tensor([[3, 4, 5], [6, 7, 8]])
C = torch.tensor(7)
D = torch.tensor([1, 2, 3])
E = torch.tensor([[4], [5], [6]])
result = torch.block_diag(A, B, C, D, E)
A = torch.tensor([[4], [3], [2]])
B = torch.tensor([7, 6, 5])
C = torch.tensor(1)
result = torch.block_diag(torch.tensor([[4], [3], [2]]), torch.tensor([7, 6,
    5]), torch.tensor(1))
a = torch.tensor([[[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]]])
b = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
result = torch.bmm(a, b)
result = torch.bmm(torch.tensor([[[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]]]),
    torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]))
src = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
result = src.to(torch.bool)
result = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).to(torch.bool)
x = 2,
y = 3, 1
result = torch.broadcast_shapes(x, y)
result = torch.broadcast_shapes((2,), (3, 1))
x = torch.tensor([[0, 1, 2]])
y = torch.tensor([[0], [1]])
result = torch.broadcast_tensors(x, y)
y = torch.tensor([[0], [1]])
result = torch.broadcast_tensors(torch.tensor([[0, 1, 2]]), y)
x = torch.tensor([1, 2, 3])
result = torch.broadcast_to(x, (3, 3))
x = torch.tensor([1, 2, 3])
shape = [3, 3]
result = torch.broadcast_to(input=x, size=shape)
boundaries = torch.tensor([1, 3, 5, 7, 9])
v = torch.tensor([[3, 6, 9], [3, 6, 9]])
result = torch.bucketize(v, boundaries)
boundaries = torch.tensor([1, 3, 5, 7, 9])
v = torch.tensor([[3, 6, 9], [3, 6, 9]])
result = torch.bucketize(input=v, boundaries=boundaries, right=True)
a = torch.tensor([1, 2, 3])
b = torch.tensor([5, 6])
result = torch.cartesian_prod(a, b)
a = torch.tensor([1, 2, 3])
result = torch.cartesian_prod(a)
x = torch.zeros(2, 3)
result = torch.cat((x, x, x))
x = torch.zeros(2, 3)
y = torch.zeros(2, 3)
result = torch.cat((x, y), 0)
x1 = torch.tensor([[1.683, 0.0526], [-0.0696, 0.6366], [-1.0091, 1.3363]])
x2 = torch.tensor([[-0.0629, 0.2414], [-0.9701, -0.4455]])
result = torch.cdist(x1, x2)
x1 = torch.tensor([[1.683, 0.0526], [-0.0696, 0.6366], [-1.0091, 1.3363]])
x2 = torch.tensor([[-0.0629, 0.2414], [-0.9701, -0.4455]])
result = torch.cdist(x1=x1, x2=x2, p=1.0, compute_mode=
    'use_mm_for_euclid_dist_if_necessary')
result = torch.ceil(torch.tensor([-0.6341, -1.4208, -1.09, 0.5826]))
input = torch.tensor([-0.6341, -1.4208, -1.09, 0.5826])
result = torch.ceil(input)
src = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
result = src.to(torch.cfloat)
result = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).to(torch.cfloat)
x = torch.ones(2, 3)
result = torch.chunk(x, 2)
result = torch.chunk(torch.ones(2, 3), chunks=2)
a = torch.tensor([-1.712, 0.1734, -0.0478, 0.8922])
result = torch.clamp(a, -0.5, 0.5)
a = torch.tensor([-1.712, 0.1734, -0.0478, 0.8922])
result = torch.clamp(a, min=-0.2, max=0.5)
x = torch.tensor([-1.712, 0.1734, -0.0478, -0.0922])
result = torch.clip(x, -0.5, 0.5)
x = torch.tensor([-1.712, 0.1734, -0.0478, -0.0922])
min, max = -0.5, 0.5
result = torch.clip(x, min, max)
real = torch.tensor([1, 2], dtype=torch.float32)
imag = torch.tensor([3, 4], dtype=torch.float32)
result = torch.complex(real, imag)
result = torch.complex(torch.tensor([1.0, 2]), torch.tensor([3.0, 4]))
src = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
result = src.to(torch.complex128)
result = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).to(torch.complex128)
src = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
result = src.to(torch.complex64)
result = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).to(torch.complex64)
x = torch.zeros(2, 3)
result = torch.concat((x, x, x))
x = torch.zeros(2, 3)
y = torch.zeros(2, 3)
result = torch.concat((x, y), 0)
x = torch.zeros(2, 3)
result = torch.concatenate((x, x, x))
x = torch.zeros(2, 3)
y = torch.zeros(2, 3)
result = torch.concatenate((x, y), 0)
x = torch.tensor([-1 + 1.0j, -2 + 2.0j, 3 - 3.0j])
result = torch.conj(x)
result = torch.conj(torch.tensor([-1 + 1.0j, -2 + 2.0j, 3 - 3.0j]))
x = torch.randn(33, 16, 30)
weight = torch.randn(20, 16, 5)
result = torch.conv1d(x, weight)
x = torch.randn(33, 16, 30)
weight = torch.randn(20, 16, 5)
bias = torch.randn(20)
result = torch.conv1d(x, weight, bias)
x = torch.randn(33, 16, 30, 30)
weight = torch.randn(20, 16, 5, 5)
result = torch.conv2d(x, weight)
x = torch.randn(33, 16, 30, 30)
weight = torch.randn(20, 16, 5, 5)
bias = torch.randn(20)
result = torch.conv2d(x, weight, bias)
x = torch.randn(33, 16, 30, 30, 30)
weight = torch.randn(20, 16, 5, 5, 5)
result = torch.conv3d(x, weight)
x = torch.randn(33, 16, 10, 10, 10)
weight = torch.randn(20, 16, 2, 2, 2)
bias = torch.randn(20)
result = torch.conv3d(x, weight, bias)
a = torch.tensor([1, 2, 3])
result = torch.copysign(a, -1, out=None)
a = torch.tensor([1, 2, 3, 4])
b = torch.tensor([-1, 2, -3, 4])
result = torch.copysign(a, b, out=None)
result = torch.cos(torch.tensor([1.4309, 1.2706, -0.8562, 0.9796]))
a = torch.tensor([1.4309, 1.2706, -0.8562, 0.9796])
result = torch.cos(a)
result = torch.cosh(torch.tensor([1.4309, 1.2706, -0.8562, 0.9796]))
a = torch.tensor([1.4309, 1.2706, -0.8562, 0.9796])
result = torch.cosh(a)
x = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
y = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
result = torch.cross(x, y)
x = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
y = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
result = torch.cross(x, y, 1)
result = torch.cuda.BFloat16Tensor(3, 5)
shape = [2, 2]
result = torch.cuda.BFloat16Tensor(*shape)
result = torch.cuda.BoolTensor(2, 3)
shape = [2, 3]
result = torch.cuda.BoolTensor(*shape)
result = torch.cuda.ByteTensor(2, 3)
shape = [2, 3]
result = torch.cuda.ByteTensor(*shape)
result = torch.cuda.CharTensor(2, 3)
shape = [2, 3]
result = torch.cuda.CharTensor(*shape)
result = torch.cuda.DoubleTensor(2, 3)
shape = [2, 3]
result = torch.cuda.DoubleTensor(*shape)
result = torch.cuda.Event(enable_timing=True)
result = torch.cuda.Event(True, interprocess=False)
result = torch.cuda.FloatTensor(2, 3)
shape = [2, 3]
result = torch.cuda.FloatTensor(*shape)
result = torch.cuda.HalfTensor(2, 3)
shape = [2, 3]
result = torch.cuda.HalfTensor(*shape)
result = torch.cuda.IntTensor(2, 3)
shape = [2, 3]
result = torch.cuda.IntTensor(*shape)
result = torch.cuda.LongTensor(2, 3)
shape = [2, 3]
result = torch.cuda.LongTensor(*shape)
result = torch.cuda.ShortTensor(2, 3)
shape = [2, 3]
result = torch.cuda.ShortTensor(*shape)
stream = torch.cuda.Stream()
result = stream.query()
stream = torch.cuda.Stream(priority=0)
result = stream.query()
s1 = torch.cuda.Stream(device=0)
a = torch.zeros(10, 10, device='cuda')
b = torch.zeros(10, 10, device='cuda')
with torch.cuda.StreamContext(stream=s1):
    result = a + b
s1 = torch.cuda.Stream(device=0)
a = torch.zeros(10, 10, device='cuda')
b = torch.zeros(10, 10, device='cuda')
with torch.cuda.StreamContext(s1):
    result = a + b
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
with torch.cuda.amp.autocast(enabled=False):
    result = x * x
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
with torch.cuda.amp.autocast():
    result = x * x
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
with torch.cuda.amp.autocast_mode.autocast(enabled=False):
    result = x * x
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
with torch.cuda.amp.autocast_mode.autocast():
    result = x * x
torch.cuda.check_error(0)
try:
    torch.cuda.check_error(1)
except RuntimeError as e:
    result1 = str(e)
try:
    torch.cuda.check_error(2)
except RuntimeError as e:
    result2 = str(e)
result = torch.cuda.cudart()
rt = torch.cuda.cudart()
result = rt.cudaMemGetInfo(0)
torch.cuda.set_device(0)
result = torch.cuda.current_device()
torch.cuda.set_device('cuda:0')
result = torch.cuda.current_device()
result = torch.cuda.current_stream()
result = torch.cuda.current_stream(0)
with torch.cuda.device(0):
    result = torch.cuda.current_device()
with torch.cuda.device(device=1):
    result = torch.cuda.current_device()
result = torch.cuda.device_count()
result = torch.cuda.empty_cache()
result = torch.cuda.get_device_capability(0)
result = torch.cuda.get_device_capability()
current_device = torch.cuda.current_device()
result = torch.cuda.get_device_name(current_device)
result = torch.cuda.get_device_name()
result = torch.cuda.get_device_properties(torch.device(0))
result = torch.cuda.get_device_properties(device='cuda:0')
torch.cuda.get_rng_state()
torch.cuda.get_rng_state(device='cuda')
result = torch.cuda.ipc_collect()
result = torch.cuda.is_available()
result = torch.cuda.is_bf16_supported()
result = torch.cuda.is_bf16_supported(including_emulation=True)
result = torch.cuda.is_current_stream_capturing()
torch.tensor([1], device='cuda:0')
result = torch.cuda.is_initialized()
x = torch.ones(2, 2).cuda()
result = torch.cuda.is_initialized()
torch.cuda.manual_seed(123)
result = paddle.get_cuda_rng_state()[0].current_seed()
torch.cuda.manual_seed(seed=123)
result = paddle.get_cuda_rng_state()[0].current_seed()
torch.cuda.manual_seed_all(123)
result = paddle.get_cuda_rng_state()[0].current_seed()
torch.cuda.manual_seed_all(seed=123)
result = paddle.get_cuda_rng_state()[0].current_seed()
result = torch.cuda.max_memory_allocated()
t = torch.tensor([1, 2, 3]).cuda()
result = torch.cuda.max_memory_allocated()
result = torch.cuda.max_memory_reserved()
t = torch.tensor([1, 2, 3]).cuda()
result = torch.cuda.max_memory_reserved()
result = torch.cuda.mem_get_info()
t = torch.tensor([1, 2, 3]).cuda()
result = torch.cuda.mem_get_info()
result = torch.cuda.memory_allocated()
t = torch.tensor([1, 2, 3]).cuda()
result = torch.cuda.memory_allocated()
a = torch.tensor([1, 2]).cuda()
result = torch.cuda.memory_reserved()
t = torch.tensor([1, 2, 3]).cuda()
result = torch.cuda.memory_reserved()
result = torch.cuda.nvtx.range_pop()
result = torch.cuda.nvtx.range_push('msg')
result = torch.cuda.nvtx.range_push(msg='msg')
result = torch.cuda.reset_peak_memory_stats()
t = torch.tensor([1, 2, 3]).cuda()
result = torch.cuda.reset_peak_memory_stats(0)
torch.cuda.set_device('cuda:1')
result = torch.cuda.current_device()
torch.cuda.set_device(device=1)
result = torch.cuda.current_device()
state = torch.cuda.get_rng_state(device='cuda')
rand1 = torch.rand([2, 3], device='cuda')
torch.cuda.set_rng_state(state, device='cuda')
rand2 = torch.rand([2, 3], device='cuda')
result = rand2 - rand1
stream = torch.cuda.Stream(0)
result = torch.cuda.set_stream(stream)
stream = torch.cuda.Stream(torch.device('cuda:0'))
result = torch.cuda.set_stream(stream=stream)
data1 = torch.ones(size=[20])
data2 = torch.ones(size=[20])
s = torch.cuda.Stream()
context = torch.cuda.stream(stream=s)
with context:
    result = data1 + data2
data1 = torch.ones(size=[20])
data2 = torch.ones(size=[20])
context = torch.cuda.stream(stream=None)
with context:
    result = data1 + data2
result = torch.cuda.synchronize(0)
t = torch.tensor([1, 2, 3]).cuda()
result = torch.cuda.synchronize(device=0)
x = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
result = torch.cumprod(x, 0)
x = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
result = torch.cumprod(x, 1, dtype=torch.float64)
x = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
result = torch.cumsum(x, 0)
x = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
result = torch.cumsum(x, dim=1)
a = torch.tensor([[180.0, -180.0], [360.0, -360.0], [90.0, -90.0]])
result = torch.deg2rad(a)
result = torch.deg2rad(torch.tensor([[180.0, -180.0], [360.0, -360.0], [
    90.0, -90.0]]))
result = torch.device('{}'.format('cpu'))
a = 'cpu'
result = torch.device(a)
x = torch.tensor([[-0.4264, 0.0255, -0.1064], [0.8795, -0.2429, 0.1374], [
    0.1029, -0.6482, -1.63]])
result = torch.diag(x, 0)
x = torch.tensor([0.595, -0.0872, 2.3298])
result = torch.diag(x)
x = torch.tensor([[0.7545889, -0.25074545, 0.5929117], [-0.6097662, -
    0.01753256, 0.619769]])
result = torch.diag_embed(x)
x = torch.tensor([[0.7545889, -0.25074545, 0.5929117], [-0.6097662, -
    0.01753256, 0.619769]])
result = torch.diag_embed(x, 1)
x = torch.tensor([[0.7545889, -0.25074545, 0.5929117], [-0.6097662, -
    0.01753256, 0.619769]])
result = torch.diagonal(x)
x = torch.tensor([[0.7545889, -0.25074545, 0.5929117], [-0.6097662, -
    0.01753256, 0.619769]])
result = torch.diagonal(x, 1)
x = torch.tensor([1, 3, 2])
result = torch.diff(x)
x = torch.tensor([1, 3, 2])
b = torch.tensor([4, 5])
result = torch.diff(x, append=b)
input = torch.tensor([-1.5393, -0.8675, 0.5916, 1.6321])
other = torch.tensor([0.0967, -1.0511, 0.6295, 0.836])
result = torch.dist(input, other, 2)
input = torch.tensor([-1.5393, -0.8675, 0.5916, 1.6321])
other = torch.tensor([0.0967, -1.0511, 0.6295, 0.836])
result = torch.dist(input, other, p=2.5)
result = torch.distributed.is_available()
result = torch.distributed.is_initialized()
a = torch.tensor([0.595, -0.0872, 2.3298, -0.2972])
result = torch.div(a, 0.5)
a = torch.tensor([[0.595, -0.0872], [2.3298, -0.2972]])
b = torch.tensor([0.1815, -1.0111])
result = torch.div(a, b)
a = torch.tensor([0.595, -0.0872, 2.3298, -0.2972])
result = torch.divide(a, 0.5)
a = torch.tensor([[0.595, -0.0872], [2.3298, -0.2972]])
b = torch.tensor([0.1815, -1.0111])
result = torch.divide(a, b)
result = torch.dot(torch.tensor([2, 3]), torch.tensor([2, 1]))
x = torch.tensor([2, 3])
y = torch.tensor([2, 1])
result = torch.dot(x, y)
src = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
result = src.to(torch.double)
result = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).to(torch.double)
result = torch.tensor([1, 2, 3], dtype=torch.float16)
result = torch.tensor([1, 0, 3], dtype=torch.bool)
result = torch.e
x = torch.tensor([[1, 2, 3], [6, 2, 9], [1, 2, 3]])
result = torch.einsum('ii->i', x)
x = torch.tensor([[1, 2, 3], [6, 2, 9]])
result = torch.einsum('ij->ji', x)
result = torch.empty(3)
result = torch.empty(3, 5)
input = torch.empty((2, 3), dtype=torch.int32)
result = torch.empty_like(input)
result = torch.empty_like(torch.empty(2, 3))
x = torch.tensor([1, 2, 3])


@torch.enable_grad()
def doubler(x):
    return x * 2


with torch.no_grad():
    result = doubler(x)
x = torch.tensor([1, 2, 3])
with torch.enable_grad():
    result = x ** 2
result = torch.eq(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4,
    4]]))
input = torch.tensor([[1, 2], [3, 4]])
other = torch.tensor([[1, 1], [4, 4]])
result = torch.eq(input, other)
result = torch.exp(torch.tensor([0.0, -2.0, 3.0]))
a = torch.tensor([-1.0, -2.0, 3.0])
result = torch.exp(a)
result = torch.expm1(torch.tensor([0.0, -2.0, 3.0]))
a = torch.tensor([-1.0, -2.0, 3.0])
result = torch.expm1(a)
result = torch.eye(3)
result = torch.eye(3, 5)
t = torch.arange(5)
result = torch.fft.fft(t)
t = torch.arange(5)
result = torch.fft.fft(input=t, n=2)
t = torch.tensor([[3.0 + 3.0j, 2.0 + 2.0j, 3.0 + 3.0j], [2.0 + 2.0j, 2.0 + 
    2.0j, 3.0 + 3.0j]])
result = torch.fft.fft2(t)
t = torch.tensor([[3.0 + 3.0j, 2.0 + 2.0j, 3.0 + 3.0j], [2.0 + 2.0j, 2.0 + 
    2.0j, 3.0 + 3.0j]])
result = torch.fft.fft2(t, s=(2, 3))
result = torch.fft.fftfreq(5)
result = torch.fft.fftfreq(n=5, d=2)
t = torch.tensor([[3.0 + 3.0j, 2.0 + 2.0j, 3.0 + 3.0j], [2.0 + 2.0j, 2.0 + 
    2.0j, 3.0 + 3.0j]])
result = torch.fft.fftn(t)
t = torch.tensor([[3.0 + 3.0j, 2.0 + 2.0j, 3.0 + 3.0j], [2.0 + 2.0j, 2.0 + 
    2.0j, 3.0 + 3.0j]])
result = torch.fft.fftn(t, s=(2, 3))
t = torch.tensor([0.0, 0.25, -0.5, -0.25])
result = torch.fft.fftshift(t)
t = torch.tensor([0.0, 0.25, -0.5, -0.25])
result = torch.fft.fftshift(t, dim=(0,))
t = torch.arange(5)
t = torch.linspace(0, 1, 5)
T = torch.fft.ifft(t)
result = torch.fft.hfft(T[:3], n=5)
t = torch.arange(5)
t = torch.linspace(0, 1, 5)
T = torch.fft.ifft(t)
result = torch.fft.hfft(T[:3])
t = torch.tensor([[3.0 + 3.0j, 2.0 + 2.0j, 3.0 + 3.0j], [2.0 + 2.0j, 2.0 + 
    2.0j, 3.0 + 3.0j]])
result = torch.fft.hfft2(t, s=(2, 5))
t = torch.tensor([[3.0 + 3.0j, 2.0 + 2.0j, 3.0 + 3.0j], [2.0 + 2.0j, 2.0 + 
    2.0j, 3.0 + 3.0j]])
result = torch.fft.hfft2(t)
t = torch.tensor([[3.0 + 3.0j, 2.0 + 2.0j, 3.0 + 3.0j], [2.0 + 2.0j, 2.0 + 
    2.0j, 3.0 + 3.0j]])
result = torch.fft.hfftn(t, s=(2, 5))
t = torch.tensor([[3.0 + 3.0j, 2.0 + 2.0j, 3.0 + 3.0j], [2.0 + 2.0j, 2.0 + 
    2.0j, 3.0 + 3.0j]])
result = torch.fft.hfftn(t)
t = torch.arange(5)
result = torch.fft.ifft(t)
t = torch.arange(5)
result = torch.fft.ifft(input=t, n=2)
t = torch.tensor([[3.0 + 3.0j, 2.0 + 2.0j, 3.0 + 3.0j], [2.0 + 2.0j, 2.0 + 
    2.0j, 3.0 + 3.0j]])
result = torch.fft.ifft2(t)
t = torch.tensor([[3.0 + 3.0j, 2.0 + 2.0j, 3.0 + 3.0j], [2.0 + 2.0j, 2.0 + 
    2.0j, 3.0 + 3.0j]])
result = torch.fft.ifft2(t, s=(2, 3))
t = torch.tensor([[3.0 + 3.0j, 2.0 + 2.0j, 3.0 + 3.0j], [2.0 + 2.0j, 2.0 + 
    2.0j, 3.0 + 3.0j]])
result = torch.fft.ifftn(t)
t = torch.tensor([[3.0 + 3.0j, 2.0 + 2.0j, 3.0 + 3.0j], [2.0 + 2.0j, 2.0 + 
    2.0j, 3.0 + 3.0j]])
result = torch.fft.ifftn(t, s=(2, 3))
t = torch.tensor([0.0, 0.25, -0.5, -0.25])
result = torch.fft.ifftshift(t)
t = torch.tensor([0.0, 0.25, -0.5, -0.25])
result = torch.fft.ifftshift(t, dim=(0,))
t = torch.arange(5)
result = torch.fft.ihfft(t)
t = torch.arange(5)
result = torch.fft.ihfft(input=t, n=2)
t = torch.arange(20).reshape((4, 5)).astype(torch.float64)
result = torch.fft.ihfft2(t, s=(2, 5))
t = torch.arange(20).reshape((4, 5)).astype(torch.float64)
result = torch.fft.ihfft2(t)
t = torch.arange(20).reshape((4, 5)).astype(torch.float64)
result = torch.fft.ihfftn(t, s=(2, 5))
t = torch.arange(20).reshape((4, 5)).astype(torch.float64)
result = torch.fft.ihfftn(t)
t = torch.tensor([[3.0 + 3.0j, 2.0 + 2.0j, 3.0 + 3.0j], [2.0 + 2.0j, 2.0 + 
    2.0j, 3.0 + 3.0j]])
result = torch.fft.irfft(t)
t = torch.tensor([[3.0 + 3.0j, 2.0 + 2.0j, 3.0 + 3.0j], [2.0 + 2.0j, 2.0 + 
    2.0j, 3.0 + 3.0j]])
result = torch.fft.irfft(t, n=1)
t = torch.tensor([[3.0 + 3.0j, 2.0 + 2.0j, 3.0 + 3.0j], [2.0 + 2.0j, 2.0 + 
    2.0j, 3.0 + 3.0j]])
result = torch.fft.irfft2(t)
t = torch.tensor([[3.0 + 3.0j, 2.0 + 2.0j, 3.0 + 3.0j], [2.0 + 2.0j, 2.0 + 
    2.0j, 3.0 + 3.0j]])
result = torch.fft.irfft2(t, s=(2, 3))
t = torch.Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
T = torch.fft.rfftn(t)
result = torch.fft.irfftn(T)
t = torch.Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
T = torch.fft.rfftn(t)
result = torch.fft.irfftn(T, norm='forward')
t = torch.Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
result = torch.fft.rfft(t)
t = torch.Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
result = torch.fft.rfft(t, n=2)
t = torch.Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
result = torch.fft.rfft2(t)
t = torch.Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
result = torch.fft.rfft2(t, s=(4, 2))
result = torch.fft.rfftfreq(5)
result = torch.fft.rfftfreq(n=5, d=2)
t = torch.Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
T = torch.fft.rfftn(t)
result = torch.fft.irfftn(T)
t = torch.Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
T = torch.fft.rfftn(t, s=(2,))
result = torch.fft.irfftn(T)
bits = torch.finfo(torch.float16).bits
min = torch.finfo(torch.float16).min
max = torch.finfo(torch.float16).max
x = torch.float32
bits = torch.finfo(x).bits
min = torch.finfo(x).min
max = torch.finfo(x).max
t = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
result = torch.flatten(t)
t = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
result = torch.flatten(t, start_dim=1)
x = torch.tensor([[0, 1], [2, 3]])
result = torch.flip(x, (0, 1))
x = torch.tensor([[0, 1], [2, 3]])
result = torch.flip(x, [0, 1])
src = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
result = src.to(torch.float16)
result = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).to(torch.float16)
src = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
result = src.to(torch.float32)
result = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).to(torch.float32)
src = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
result = src.to(torch.float64)
result = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).to(torch.float64)
src = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
result = src.to(torch.float8_e4m3fn).float()
result = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).to(torch.float8_e4m3fn
    ).float()
input = torch.tensor([-0.8166, 1.5308, -0.253, -0.2091])
result = torch.floor(input)
result = torch.floor(torch.tensor([-0.8166, 1.5308, -0.253, -0.2091]))
a = torch.tensor([4.0, 3.0])
b = torch.tensor([2.0, 2.0])
result = torch.floor_divide(a, b)
result = torch.floor_divide(torch.tensor([4.0, 3.0]), torch.tensor([2.0, 2.0]))
result = torch.fmax(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [
    4, 4]]))
input = torch.tensor([[1, 2], [3, 4]])
other = torch.tensor([[1, 1], [4, 4]])
result = torch.fmax(input, other)
result = torch.fmin(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [
    4, 4]]))
input = torch.tensor([[1, 2], [3, 4]])
other = torch.tensor([[1, 1], [4, 4]])
result = torch.fmin(input, other)
result = torch.frac(torch.tensor([1, 2.5, -3.2]))
a = torch.tensor([1, 2.5, -3.2])
result = torch.frac(a)
a = numpy.array([1, 2, 3])
result = torch.from_numpy(a)
result = torch.from_numpy(numpy.array([1, 2, 3]))
input = torch.empty(2, 3)
result = torch.full(input.shape, 2.0)
num = 5.0
result = torch.full((2, 3), num)
input = torch.empty(2, 3)
result = torch.full_like(input, 2)
num = 5.0
result = torch.full_like(torch.empty(2, 3), num)
result = torch.functional.atleast_1d(torch.tensor(123, dtype=torch.int32))
y = torch.tensor([-1, -2, 3])
result = torch.functional.atleast_1d((torch.tensor(123, dtype=torch.int32), y))
result = torch.functional.atleast_2d(torch.tensor(123, dtype=torch.int32))
y = torch.tensor([-1, -2, 3])
result = torch.functional.atleast_2d((torch.tensor(123, dtype=torch.int32), y))
result = torch.functional.atleast_3d(torch.tensor(123, dtype=torch.int32))
y = torch.tensor([-1, -2, 3])
result = torch.functional.atleast_3d((torch.tensor(123, dtype=torch.int32), y))
x = 2,
y = 3, 1
result = torch.functional.broadcast_shapes(x, y)
result = torch.functional.broadcast_shapes((2,), (3, 1))
x = torch.tensor([[1, 2, 3], [6, 2, 9], [1, 2, 3]])
result = torch.functional.einsum('ii->i', x)
x = torch.tensor([[1, 2, 3], [6, 2, 9]])
result = torch.functional.einsum('ij->ji', x)
x = torch.tensor([1, 2, 3])
y = torch.tensor([3, 4, 5])
grid_x, grid_y = torch.functional.meshgrid(x, y, indexing='ij')
result = grid_x
x = torch.tensor([1, 2, 3])
y = torch.tensor([3, 4, 5])
grid_x, grid_y = torch.functional.meshgrid(x, y, indexing='ij')
result = grid_y
input = torch.tensor([[[-12.0, -11.0, -10.0, -9.0], [-8.0, -7.0, -6.0, -5.0
    ], [-4.0, -3.0, -2.0, -1.0]], [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 
    7.0], [8.0, 9.0, 10.0, 11.0]]])
result = torch.functional.norm(input, p='fro')
input = torch.tensor([[-12.0, -11.0, -10.0, -9.0], [-8.0, -7.0, -6.0, -5.0],
    [-4.0, -3.0, -2.0, -1.0]])
result = torch.functional.norm(input, p='nuc')
a = torch.arange(12).reshape(6, 2)
result = torch.functional.split(a, 2)
a = torch.arange(12).reshape(6, 2)
result = torch.functional.split(a, 2, dim=0)
x = torch.tensor([1, 1, 2, 2, 3, 1, 1, 2])
result = torch.functional.unique_consecutive(x)
x = torch.tensor([1, 1, 2, 2, 3, 1, 1, 2])
result, inverse_indices = torch.functional.unique_consecutive(x,
    return_inverse=True)
a = torch.tensor([[1, 2], [3, 4]])
result = torch.gather(a, 1, torch.tensor([[0, 0], [1, 0]]))
result = torch.gather(torch.tensor([[1, 2], [3, 4]]), 1, torch.tensor([[0, 
    0], [1, 0]]))
result = torch.ge(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4,
    4]]))
input = torch.tensor([[1, 2], [3, 4]])
other = torch.tensor([[1, 1], [4, 4]])
result = torch.ge(input, other)
x = torch.tensor([1.0, 2, 3])
y = torch.tensor([1.0, 2, 3, 4])
result = torch.ger(x, y)
x = torch.tensor([1.0, 2, 3])
y = torch.tensor([1.0, 2, 3, 4])
result = torch.ger(input=x, vec2=y)
result_gpu = torch.get_autocast_gpu_dtype()
result_before = torch.get_autocast_gpu_dtype()
with torch.autocast('cuda', dtype=torch.bfloat16):
    result_inside = torch.get_autocast_gpu_dtype()
result_after = torch.get_autocast_gpu_dtype()
paddle.device.set_device(device=device2str('cpu'))
result = torch.get_default_device()
paddle.device.set_device(device=device2str(None))
paddle.device.set_device(device=device2str(torch.device('cpu:1')))
result = torch.get_default_device()
paddle.device.set_device(device=device2str(None))
result = torch.get_default_dtype()
t = torch.tensor([1, 2, 3]).cuda()
result = torch.get_device(t)
result = torch.get_device_module('cuda')
result = torch.get_device_module(torch.device('cuda'))
torch.get_rng_state()
result = torch.greater(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1],
    [4, 4]]))
input = torch.tensor([[1, 2], [3, 4]])
other = torch.tensor([[1, 1], [4, 4]])
result = torch.greater(input, other)
result = torch.greater_equal(torch.tensor([[1, 2], [3, 4]]), torch.tensor([
    [1, 1], [4, 4]]))
input = torch.tensor([[1, 2], [3, 4]])
other = torch.tensor([[1, 1], [4, 4]])
result = torch.greater_equal(input, other)
x = torch.tensor([[[[-1.2392, -0.131, -0.6679, 0.5476], [1.1738, -1.7384, -
    0.7733, 0.3261], [-0.0926, -1.0448, -1.2557, -1.5503], [0.6402, 0.9072,
    0.678, -1.9885]], [[0.0639, -1.1592, 1.4242, -0.4641], [-0.192, 0.1826,
    1.9217, -0.4359], [1.1926, -0.0247, 0.4744, -1.0216], [-0.036, -1.1656,
    0.3661, -1.8147]]]])
result = torch.group_norm(x, 2)
x = torch.tensor([[[[-1.2392, -0.131, -0.6679, 0.5476], [1.1738, -1.7384, -
    0.7733, 0.3261], [-0.0926, -1.0448, -1.2557, -1.5503], [0.6402, 0.9072,
    0.678, -1.9885]], [[0.0639, -1.1592, 1.4242, -0.4641], [-0.192, 0.1826,
    1.9217, -0.4359], [1.1926, -0.0247, 0.4744, -1.0216], [-0.036, -1.1656,
    0.3661, -1.8147]]]])
result = torch.group_norm(x, 2)
result = torch.gt(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4,
    4]]))
input = torch.tensor([[1, 2], [3, 4]])
other = torch.tensor([[1, 1], [4, 4]])
result = torch.gt(input, other)
result = torch.hamming_window(10)
result = torch.hamming_window(10, dtype=torch.float64)
input = torch.tensor([-1.5, 0, 2.0])
values = torch.tensor([0.5])
result = torch.heaviside(input, values)
input = torch.tensor([-1.5, 0, 2.0])
values = torch.tensor([0.5])
out = torch.tensor([-1.5, 0, 2.0])
torch.heaviside(input, values, out=out)
result = torch.special.i0(torch.tensor([1.0, 1.2661, 2.2796]))
a = torch.tensor([1.0, 1.2661, 2.2796])
result = torch.special.i0(a)
x = torch.ones([5, 3])
t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
index = torch.tensor([0, 4, 2])
result = torch.index_add(x, 0, index, t)
x = torch.ones([5, 3])
t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
index = torch.tensor([0, 4, 2])
result = torch.index_add(input=x, dim=0, index=index, source=t)
x = torch.eye(2, 4)
indices = torch.tensor([0, 1])
value = -1
result = torch.index_fill(x, 0, indices, value)
indices = torch.tensor([0, 1])
value = -1
result = torch.index_fill(torch.eye(3, 4), 1, indices, value)
x = torch.ones([5, 3])
t = torch.tensor([1.0], dtype=torch.float)
indices = [torch.tensor(i) for i in [[0, 0], [0, 1]]]
result = torch.index_put(x, indices, t)
x = torch.ones([5, 3])
t = torch.tensor([1.0], dtype=torch.float)
indices = [torch.tensor(i) for i in [[0, 0], [0, 1]]]
result = torch.index_put(x, indices, values=t)
x = torch.eye(2, 4)
indices = torch.tensor([0, 1])
result = torch.index_select(x, 0, indices)
indices = torch.tensor([0, 1])
result = torch.index_select(torch.eye(3, 4), 1, indices)
result = torch.inf
src = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
result = src.to(torch.int16)
result = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).to(torch.int16)
src = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
result = src.to(torch.int32)
result = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).to(torch.int32)
src = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
result = src.to(torch.int64)
result = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).to(torch.int64)
src = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
result = src.to(torch.int8)
result = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).to(torch.int8)
x = torch.tensor([[0.7308, 1.006, 0.527, 1.4516], [-0.1383, 1.5706, 0.4724,
    0.4141], [0.1193, 0.2829, 0.9037, 0.3957], [-0.8202, -0.6474, -0.1631, 
    -0.6543]])
result = torch.inverse(x)
x = torch.tensor([[[[-0.1533, 2.302, -0.1771, 0.5928], [0.4338, -0.6537, 
    0.2296, 0.5946], [-0.4932, 1.8386, -0.1039, 1.044], [0.1735, -0.8303, -
    0.3821, -0.4384]], [[-0.1533, 2.302, -0.1771, 0.5928], [0.4338, -0.6537,
    0.2296, 0.5946], [-0.4932, 1.8386, -0.1039, 1.044], [0.1735, -0.8303, -
    0.3821, -0.4384]]]])
result = torch.inverse(x)
result = torch.is_autocast_enabled()
result_before = torch.is_autocast_enabled()
with torch.autocast(device_type='cuda', enabled=True):
    result_inside = torch.is_autocast_enabled()
result_after = torch.is_autocast_enabled()
a = torch.tensor([[4, 9], [23, 2]])
result = torch.is_complex(a)
>>>>>>result = torch.is_complex(torch.tensor([[4, 9], [23, 2]], dtype=torch.
    complex64))
a = torch.tensor([[4, 9], [23, 2]], dtype=torch.int64)
result = torch.is_floating_point(a)
a = torch.tensor([[4, 9], [23, 2]], dtype=torch.float64)
result = torch.is_floating_point(a)
result = torch.is_grad_enabled()
a = torch.tensor([[4, 9], [23, 2]])
result = torch.is_tensor(a)
result = torch.is_tensor(torch.tensor([[4, 9], [23, 2]]))
result = torch.isclose(torch.tensor([10000.0, 1e-07]), torch.tensor([
    10000.1, 1e-08]))
result = torch.isclose(torch.tensor([10000.0, 1e-08]), torch.tensor([
    10000.1, 1e-09]))
result = torch.isfinite(torch.tensor([1, float('inf'), 2, float('-inf'),
    float('nan')]))
input = torch.tensor([1, float('inf'), 2, float('-inf'), float('nan')])
result = torch.isfinite(input)
result = torch.isin(torch.tensor([[1, 2], [3, 4]]), torch.tensor([2, 3]))
result = torch.isin(torch.tensor([[1, 2], [3, 4]]), torch.tensor([2, 3]),
    assume_unique=True)
result = torch.isinf(torch.tensor([1, float('inf'), 2, float('-inf'), float
    ('nan')]))
input = torch.tensor([1, float('inf'), 2, float('-inf'), float('nan')])
result = torch.isinf(input)
result = torch.isnan(torch.tensor([1, float('inf'), 2, float('-inf'), float
    ('nan')]))
input = torch.tensor([1, float('inf'), 2, float('-inf'), float('nan')])
result = torch.isnan(input)
input = torch.tensor([[[1.1524, 0.4714, 0.2857], [-1.2533, -0.9829, -1.0981
    ], [0.1507, -1.1431, -2.0361]], [[0.1024, -0.4482, 0.4137], [0.9385, 
    0.4565, 0.7702], [0.4135, -0.2587, 0.0482]]])
data = torch.tensor([1.0, 1.0, 1.0])
result = torch.layer_norm(input, [3], data, data)
input = torch.tensor([[[1.1524, 0.4714, 0.2857], [-1.2533, -0.9829, -1.0981
    ], [0.1507, -1.1431, -2.0361]], [[0.1024, -0.4482, 0.4137], [0.9385, 
    0.4565, 0.7702], [0.4135, -0.2587, 0.0482]]])
data = torch.tensor([1.0, 1.0, 1.0])
result = torch.layer_norm(input=input, normalized_shape=[3], weight=data,
    bias=data)
result = torch.le(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4,
    4]]))
input = torch.tensor([[1, 2], [3, 4]])
other = torch.tensor([[1, 1], [4, 4]])
result = torch.le(input, other)
result = torch.lerp(torch.tensor([1.0, 2.0, 3.0, 4.0]), torch.tensor([10.0,
    10.0, 10.0, 10.0]), 0.5)
>>>>>>result = torch.lerp(input=torch.tensor([1.0, 2.0, 3.0, 4.0]), end=torch.
    tensor([10.0, 10.0, 10.0, 10.0]), weight=0.5)
result = torch.less(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [
    4, 4]]))
input = torch.tensor([[1, 2], [3, 4]])
other = torch.tensor([[1, 1], [4, 4]])
result = torch.less(input, other)
result = torch.less_equal(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1,
    1], [4, 4]]))
input = torch.tensor([[1, 2], [3, 4]])
other = torch.tensor([[1, 1], [4, 4]])
result = torch.less_equal(input, other)
x = torch.tensor([[4.0, 5.0, 6.0], [1.0, 2.0, 3.0], [4.0, 9.0, 10.0]])
y = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
result = torch.linalg.matmul(x, y)
x = torch.tensor([[4.0, 5.0, 6.0], [1.0, 2.0, 3.0], [4.0, 9.0, 10.0]])
y = torch.tensor([1.0, 2.0, 3.0])
result = torch.linalg.matmul(x, y)
y = torch.tensor([[3, 4, 6], [5, 3, 4], [1, 2, 1.0]])
result = torch.linalg.norm(y, dim=-1, keepdim=True)
x = torch.tensor([[10, 2, 3], [3, 10, 5], [5, 6, 12.0]])
result = torch.linalg.norm(input=x, ord=float('inf'), dim=-1)
x = torch.tensor([[3.0, 1], [1, 2]])
y = torch.tensor([9.0, 8])
result = torch.linalg.solve(x, y)
x = torch.tensor([[3.0, 1], [1, 2]])
y = torch.tensor([9.0, 8])
result = torch.linalg.solve(A=x, B=y)
x = torch.tensor([[0.02773777, 0.93004224, 0.06911496], [0.24831591, 
    0.45733623, 0.07717843], [0.48016702, 0.14235102, 0.42620817]])
result = torch.linalg.vector_norm(x)
x = torch.tensor([[0.02773777, 0.93004224, 0.06911496], [0.24831591, 
    0.45733623, 0.07717843], [0.48016702, 0.14235102, 0.42620817]])
result = torch.linalg.vector_norm(x=x, ord=2)
result = torch.linspace(3, 10, 5)
result = torch.linspace(-10.0, 10.0, 5)
input = torch.tensor([4.7767, 4.3234, 1.2156, 0.2411, 4.5739])
result = torch.log(input)
result = torch.log(torch.tensor([4.7767, 4.3234, 1.2156, 0.2411, 4.5739]))
input = torch.tensor([4.7767, 4.3234, 1.2156, 0.2411, 4.5739])
result = torch.log10(input)
result = torch.log10(torch.tensor([4.7767, 4.3234, 1.2156, 0.2411, 4.5739]))
input = torch.tensor([4.7767, 4.3234, 1.2156, 0.2411, 4.5739])
result = torch.log1p(input)
result = torch.log1p(torch.tensor([4.7767, 4.3234, 1.2156, 0.2411, 4.5739]))
input = torch.tensor([4.7767, 4.3234, 1.2156, 0.2411, 4.5739])
result = torch.log2(input)
result = torch.log2(torch.tensor([4.7767, 4.3234, 1.2156, 0.2411, 4.5739]))
result = torch.logical_and(torch.tensor([True, False, True]), torch.tensor(
    [True, False, False]))
a = torch.tensor([0, 1, 10, 0], dtype=torch.int8)
b = torch.tensor([4, 0, 1, 0], dtype=torch.int8)
result = torch.logical_and(a, b)
result = torch.logical_not(torch.tensor([True, False, True]))
a = torch.tensor([0, 1, 10, 0], dtype=torch.int8)
result = torch.logical_not(a)
result = torch.logical_or(torch.tensor([True, False, True]), torch.tensor([
    True, False, False]))
a = torch.tensor([0, 1, 10, 0], dtype=torch.int8)
b = torch.tensor([4, 0, 1, 0], dtype=torch.int8)
result = torch.logical_or(a, b)
result = torch.logical_xor(torch.tensor([True, False, True]), torch.tensor(
    [True, False, False]))
a = torch.tensor([0, 1, 10, 0], dtype=torch.int8)
b = torch.tensor([4, 0, 1, 0], dtype=torch.int8)
result = torch.logical_xor(a, b)
input = torch.tensor([1.4907, 1.0593, 1.5696])
result = torch.logsumexp(input, 0)
input = torch.tensor([[1.4907, 1.0593, 1.5696], [1.4907, 1.0593, 1.5696]])
result = torch.logsumexp(input, 1)
src = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
result = src.to(torch.long)
result = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).to(torch.long)
result = torch.lt(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4,
    4]]))
input = torch.tensor([[1, 2], [3, 4]])
other = torch.tensor([[1, 1], [4, 4]])
result = torch.lt(input, other)
torch.manual_seed(100)
result = paddle.get_rng_state()[0].current_seed()
torch.manual_seed(seed=100)
result = paddle.get_rng_state()[0].current_seed()
x = torch.eye(2, 4)
mask = x > 0
result = torch.masked_select(x, mask)
x = torch.ones(2, 4)
result = torch.masked_select(x, x > 0)
x = torch.tensor([[4.0, 5.0, 6.0], [1.0, 2.0, 3.0], [4.0, 9.0, 10.0]])
y = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
result = torch.matmul(x, y)
x = torch.tensor([[4.0, 5.0, 6.0], [1.0, 2.0, 3.0], [4.0, 9.0, 10.0]])
y = torch.tensor([1.0, 2.0, 3.0])
result = torch.matmul(x, y)
result = torch.maximum(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1],
    [4, 4]]))
input = torch.tensor([[1, 2], [3, 4]])
other = torch.tensor([[1, 1], [4, 4]])
result = torch.maximum(input, other)
input = torch.tensor([1.4907, 1.0593, 1.5696])
result = torch.mean(input)
input = torch.tensor([[1.4907, 1.0593, 1.5696], [1.4907, 1.0593, 1.5696]])
result = torch.mean(input, 1)
x = torch.tensor([1, 2, 3])
y = torch.tensor([3, 4, 5])
grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
result = grid_x
x = torch.tensor([1, 2, 3])
y = torch.tensor([3, 4, 5])
grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
result = grid_y
result = torch.minimum(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1],
    [4, 4]]))
input = torch.tensor([[1, 2], [3, 4]])
other = torch.tensor([[1, 1], [4, 4]])
result = torch.minimum(input, other)
a = torch.tensor([[1.0, 2.0], [4.0, 5.0]])
b = torch.tensor([[1.0, 3.0], [3.0, 6.0]])
result = torch.mm(a, b)
a = torch.tensor([[1.0, 2.0], [4.0, 5.0]])
b = torch.tensor([[1.0, 3.0], [3.0, 6.0]])
result = torch.mm(input=a, mat2=b)
x = torch.tensor([[-1.3029, 0.4921, -0.7432], [2.6672, -0.0987, 0.075], [
    0.1436, -1.0114, 1.3641]])
result = torch.msort(x)
x = torch.tensor([[-1.3029, 0.4921, -0.7432], [2.6672, -0.0987, 0.075], [
    0.1436, -1.0114, 1.3641]])
result = torch.msort(input=x)
input = torch.tensor([0.2015, -0.4255, 2.6087])
other = torch.tensor([0.2015, -0.4255, 2.6087])
result = torch.mul(input, other)
input = torch.tensor([0.2015, -0.4255, 2.6087])
other = torch.tensor([2, 6, 4])
result = torch.mul(input, other)
torch.manual_seed(100)
weights = torch.tensor([0, 10, 3, 0], dtype=torch.float)
result = torch.multinomial(weights, 2)
torch.manual_seed(100)
weights = torch.tensor([0, 10, 3, 0], dtype=torch.float)
result = torch.multinomial(weights, 4, replacement=True)
input = torch.tensor([0.2015, -0.4255, 2.6087])
other = torch.tensor([0.2015, -0.4255, 2.6087])
result = torch.multiply(input, other)
input = torch.tensor([0.2015, -0.4255, 2.6087])
other = torch.tensor([2, 6, 4])
result = torch.multiply(input, other)
result = torch.nan
x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
result = torch.narrow(x, 0, 0, 2)
x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
result = torch.narrow(x, 1, 1, 2)
result = torch.ne(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4,
    4]]))
input = torch.tensor([[1, 2], [3, 4]])
other = torch.tensor([[1, 1], [4, 4]])
result = torch.ne(input, other)
result = torch.newaxis
input = torch.tensor([1.0, 2.0, 3.0])
other = torch.tensor([1.1, 2.1, 3.1])
result = torch.nextafter(input, other)
a = torch.tensor([0.0, 1.0, 2.0])
b = torch.tensor([0.5, 1.5, 2.5])
result = torch.nextafter(a, b)
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
model = nn.AdaptiveAvgPool1d(5)
result = model(x)
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
model = nn.AdaptiveAvgPool1d(output_size=5)
result = model(x)
x = torch.tensor([[[[0.9785, 1.2013, 2.4873, -1.1891], [-0.0832, -0.5456, -
    0.5009, 1.5103], [-1.286, 1.0287, -1.3902, 0.4627], [-0.0502, -1.3924, 
    -0.3327, 0.1678]]]])
model = nn.AdaptiveAvgPool2d(5)
result = model(x)
x = torch.tensor([[[[0.9785, 1.2013, 2.4873, -1.1891], [-0.0832, -0.5456, -
    0.5009, 1.5103], [-1.286, 1.0287, -1.3902, 0.4627], [-0.0502, -1.3924, 
    -0.3327, 0.1678]]]])
model = nn.AdaptiveAvgPool2d(output_size=(2, 2))
result = model(x)
x = torch.tensor([[[[[-1.1494, -1.3829], [0.4995, -1.3094]], [[1.0015, 
    1.4919], [-1.5187, 0.0235]]]]])
model = nn.AdaptiveAvgPool3d(1)
result = model(x)
x = torch.tensor([[[[[-1.1494, -1.3829], [0.4995, -1.3094]], [[1.0015, 
    1.4919], [-1.5187, 0.0235]]]]])
model = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
result = model(x)
input = torch.tensor([[0.9368637, -0.0361056, -0.98917043, 0.06605113, 
    1.5254455], [-1.0518035, -1.0024613, 0.18699688, -0.35807893, 
    0.25628588], [-0.900478, -0.41495147, 0.84707606, -1.7883497, 1.3243382]])
target = torch.tensor([1, 1, 1])
asfm = nn.AdaptiveLogSoftmaxWithLoss(5, 4, [2])
out, loss = asfm(input, target)
input = torch.tensor([[0.9368637, -0.0361056, -0.98917043, 0.06605113, 
    1.5254455], [-1.0518035, -1.0024613, 0.18699688, -0.35807893, 
    0.25628588], [-0.900478, -0.41495147, 0.84707606, -1.7883497, 1.3243382]])
target = torch.tensor([1, 1, 1])
asfm = nn.AdaptiveLogSoftmaxWithLoss(5, 4, [3], div_value=2.0)
out, loss = asfm(input, target)
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
model = nn.AdaptiveMaxPool1d(5)
result = model(x)
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
model = nn.AdaptiveMaxPool1d(output_size=5)
result = model(x)
x = torch.tensor([[[[0.9785, 1.2013, 2.4873, -1.1891], [-0.0832, -0.5456, -
    0.5009, 1.5103], [-1.286, 1.0287, -1.3902, 0.4627], [-0.0502, -1.3924, 
    -0.3327, 0.1678]]]])
model = nn.AdaptiveMaxPool2d(5)
result = model(x)
x = torch.tensor([[[[0.9785, 1.2013, 2.4873, -1.1891], [-0.0832, -0.5456, -
    0.5009, 1.5103], [-1.286, 1.0287, -1.3902, 0.4627], [-0.0502, -1.3924, 
    -0.3327, 0.1678]]]])
model = nn.AdaptiveMaxPool2d(output_size=(2, 2))
result = model(x)
x = torch.tensor([[[[[-1.1494, -1.3829], [0.4995, -1.3094]], [[1.0015, 
    1.4919], [-1.5187, 0.0235]]]]])
model = nn.AdaptiveMaxPool3d(1)
result = model(x)
x = torch.tensor([[[[[-1.1494, -1.3829], [0.4995, -1.3094]], [[1.0015, 
    1.4919], [-1.5187, 0.0235]]]]])
model = nn.AdaptiveMaxPool3d(output_size=(1, 1, 1))
result = model(x)
x = torch.tensor([[[[0.0]], [[0.1]], [[0.2]], [[0.30000001]], [[0.40000001]
    ], [[0.5]]]])
model = nn.ChannelShuffle(3)
result = model(x)
x = torch.tensor([[[[0.0]], [[0.1]], [[0.2]], [[0.30000001]], [[0.40000001]
    ], [[0.5]]]])
model = nn.ChannelShuffle(groups=2)
result = model(x)
x = torch.tensor([[[-1.3328, -0.4948], [0.8689, 1.1423]], [[-0.2671, -
    1.0868], [1.3011, 1.0469]]])
model = nn.CircularPad1d(1)
result = model(x)
padding = model.padding
x = torch.tensor([[[-1.3328, -0.4948], [0.8689, 1.1423]], [[-0.2671, -
    1.0868], [1.3011, 1.0469]]])
model = nn.CircularPad1d((1, 1))
result = model(x)
padding = model.padding
x = torch.tensor([[[[-1.3328, -0.4948], [0.8689, 1.1423]], [[-0.2671, -
    1.0868], [1.3011, 1.0469]]]])
model = nn.CircularPad2d(1)
result = model(x)
padding = model.padding
x = torch.tensor([[[[-1.3328, -0.4948], [0.8689, 1.1423]], [[-0.2671, -
    1.0868], [1.3011, 1.0469]]]])
model = nn.CircularPad2d((1, 1, 1, 1))
result = model(x)
padding = model.padding
x = torch.tensor([[[[[-1.3328, -0.4948], [0.8689, 1.1423]], [[-0.2671, -
    1.0868], [1.3011, 1.0469]]]]])
model = nn.CircularPad3d(1)
result = model(x)
padding = model.padding
x = torch.tensor([[[[[-1.3328, -0.4948], [0.8689, 1.1423]], [[-0.2671, -
    1.0868], [1.3011, 1.0469]]]]])
model = nn.CircularPad3d((1, 1, 1, 1, 1, 1))
result = model(x)
padding = model.padding
x = torch.tensor([[[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]])
model = nn.ConstantPad1d(2, 3.5)
result = model(x)
padding = model.padding
value = model.value
x = torch.tensor([[[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]])
model = nn.ConstantPad1d((2, 1), 3.5)
result = model(x)
padding = model.padding
value = model.value
x = torch.tensor([[[[-0.4106, 0.1677], [-0.6648, -0.5669]]]])
model = nn.ConstantPad2d(1, 4.7)
result = model(x)
padding = model.padding
value = model.value
x = torch.tensor([[[[-0.4106, 0.1677], [-0.6648, -0.5669]]]])
model = nn.ConstantPad2d((1, 1, 1, 0), 4.8)
result = model(x)
padding = model.padding
value = model.value
x = torch.tensor([[[[[-1.3328, -0.4948], [0.8689, 1.1423]], [[-0.2671, -
    1.0868], [1.3011, 1.0469]]]]])
model = nn.ConstantPad3d(1, 4.5)
result = model(x)
padding = model.padding
value = model.value
x = torch.tensor([[[[[-1.3328, -0.4948], [0.8689, 1.1423]], [[-0.2671, -
    1.0868], [1.3011, 1.0469]]]]])
model = nn.ConstantPad3d((1, 1, 1, 1, 1, 1), 4.6)
result = model(x)
padding = model.padding
value = model.value
x = torch.randn(20, 16, 50)
model = nn.Conv1d(16, 33, 3, stride=2, bias=False)
result = model(x)
x = torch.randn(20, 16, 50)
model = nn.Conv1d(16, 33, 3, stride=2, padding=4, bias=False)
result = model(x)
x = torch.zeros(20, 16, 50, 100)
model = nn.Conv2d(16, 33, 3, stride=2, bias=False)
result = model(x)
x = torch.zeros(20, 16, 50, 100)
model = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), bias=False)
result = model(x)
x = torch.rand(2, 16, 50, 20, 20)
model = nn.Conv3d(16, 33, 3, stride=2, bias=False)
result = model(x)
x = torch.randn(2, 16, 50, 20, 20)
model = nn.Conv3d(16, 33, (3, 3, 5), stride=(2, 2, 1), padding=(4, 2, 2),
    bias=False)
result = model(x)
x = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
y = torch.tensor([[8.0, 3.0, 3.0], [2.0, 3.0, 4.0]])
model = nn.CosineSimilarity()
result = model(x, y)
x = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
y = torch.tensor([[8.0, 3.0, 3.0], [2.0, 3.0, 4.0]])
model = nn.CosineSimilarity(0)
result = model(x, y)
x = torch.randn(20, 16)
model = nn.Dropout(0.4)
result = model(x)
x = torch.randn(20, 16)
model = nn.Dropout(0.4, False)
result = model(x)
embedding = torch.nn.Embedding(4, 3)
w0 = torch.Tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0,
    3.0, 3.0]])
with torch.no_grad():
    embedding.weight[0] = w0[0]
    embedding.weight[1] = w0[1]
    embedding.weight[3] = w0[3]
x = torch.LongTensor([[0], [1], [3]])
result = embedding(x)
padding_idx = 0
embedding = torch.nn.Embedding(4, 3, padding_idx=padding_idx)
w0 = torch.Tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0,
    3.0, 3.0]])
with torch.no_grad():
    embedding.weight[0] = w0[0]
    embedding.weight[1] = w0[1]
    embedding.weight[2] = w0[2]
    embedding.weight[3] = w0[3]
x = torch.LongTensor([[0], [1], [3]])
result = embedding(x)
x = torch.ones([2, 3 * 2 * 2, 12])
fold = nn.Fold([4, 5], 2)
result = fold(x)
x = torch.ones([2, 3 * 2 * 2, 40])
fold = nn.Fold([4, 5], 2, 1, [1, 2], 1)
result = fold(x)
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
model = nn.GELU()
result = model(x)
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
model = nn.GELU(approximate='tanh')
result = model(x)
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
m = torch.nn.GLU()
result = m(x)
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
m = torch.nn.GLU(dim=-1)
result = m(x)
a = torch.tensor([[[[2.0, 3.0], [3.0, 5.0]], [[5.0, 3.0], [9.0, 5.0]]]])
m = torch.nn.GroupNorm(2, 2)
result = m(a)
a = torch.tensor([[[[2.0, 3.0], [3.0, 5.0]], [[5.0, 3.0], [9.0, 5.0]]]])
m = torch.nn.GroupNorm(2, 2, eps=1e-05, affine=False)
result = m(a)
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
model = nn.Hardshrink(0.8)
result = model(x)
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
model = nn.Hardshrink(0.4)
result = model(x)
input = torch.tensor([[-1.2837, -0.0297, 0.0355], [0.9112, -1.7526, -0.4061]])
target = torch.tensor([[1.0, 2.0, 1.0], [1.0, 2.0, 3.0]])
loss = torch.nn.HuberLoss()
result = loss(input, target)
input = torch.tensor([[-1.2837, -0.0297, 0.0355], [0.9112, -1.7526, -0.4061]])
target = torch.tensor([[1.0, 2.0, 1.0], [1.0, 2.0, 3.0]])
loss = torch.nn.HuberLoss(reduction='mean')
result = loss(input, target)
m = nn.Identity(20)
input = torch.ones(128, 20)
result = m(input)
m = nn.Identity(54, unused_argument1=0.1, unused_argument2=False)
input = torch.ones(128, 20)
output = m(input)
result = m(input)
input = torch.tensor([[[-0.5743, 0.4889, -0.0878, 0.421, -0.0844], [0.3614,
    0.8458, -0.6152, 0.6894, 0.2927], [-0.0087, 0.1098, 0.1783, -0.6953, 
    0.5519], [0.3789, -0.056, -0.409, -0.107, -1.0139], [0.9204, 1.0817, -
    2.6126, 0.4244, 0.3272]]])
pool = nn.LPPool1d(1, 3, stride=2)
result = pool(input)
input = torch.tensor([[[0.643, 0.4511, -1.6757, 1.7116], [-0.2288, -0.4111,
    -1.3602, 0.2685], [0.2363, 1.9341, 0.8522, -0.1846], [1.6496, -0.0675, 
    -0.7208, -1.0018]], [[-0.3183, 0.8029, -0.4993, 1.0598], [-0.4952, -
    0.9536, 0.1954, 0.0551], [1.2257, 0.7517, 0.4063, -1.2151], [-1.3562, 
    0.3547, 1.1147, 1.2898]], [[0.1205, -0.1889, 0.5086, -0.808], [0.3156, 
    -0.8298, 2.0242, -0.9184], [-0.4005, 1.3586, 0.6205, -0.7487], [1.6239,
    0.29, 0.9671, 1.2961]], [[-1.1996, -0.2201, -0.9466, -0.7264], [-0.0313,
    0.8284, -0.3588, 1.3522], [-0.0991, -0.5112, -0.1785, 2.0903], [-1.3286,
    -0.9333, -0.1404, 1.2582]]])
pool = nn.LPPool1d(2, 4, stride=2)
result = pool(input)
input = torch.tensor([[[[-0.5743, 0.4889, -0.0878, 0.421, -0.0844], [0.3614,
    0.8458, -0.6152, 0.6894, 0.2927], [-0.0087, 0.1098, 0.1783, -0.6953, 
    0.5519], [0.3789, -0.056, -0.409, -0.107, -1.0139], [0.9204, 1.0817, -
    2.6126, 0.4244, 0.3272]]]])
pool = nn.LPPool2d(1, 3, stride=2)
result = pool(input)
input = torch.tensor([[[[0.643, 0.4511, -1.6757, 1.7116], [-0.2288, -0.4111,
    -1.3602, 0.2685], [0.2363, 1.9341, 0.8522, -0.1846], [1.6496, -0.0675, 
    -0.7208, -1.0018]], [[-0.3183, 0.8029, -0.4993, 1.0598], [-0.4952, -
    0.9536, 0.1954, 0.0551], [1.2257, 0.7517, 0.4063, -1.2151], [-1.3562, 
    0.3547, 1.1147, 1.2898]], [[0.1205, -0.1889, 0.5086, -0.808], [0.3156, 
    -0.8298, 2.0242, -0.9184], [-0.4005, 1.3586, 0.6205, -0.7487], [1.6239,
    0.29, 0.9671, 1.2961]], [[-1.1996, -0.2201, -0.9466, -0.7264], [-0.0313,
    0.8284, -0.3588, 1.3522], [-0.0991, -0.5112, -0.1785, 2.0903], [-1.3286,
    -0.9333, -0.1404, 1.2582]]]])
pool = nn.LPPool2d(2, 4, stride=2)
result = pool(input)
m = nn.LayerNorm(10)
input = torch.ones(2, 5, 10)
result = m(input)
m = nn.LayerNorm([5, 10, 10])
input = torch.ones(2, 5, 10, 10)
result = m(input)
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
model = nn.LeakyReLU()
result = model(x)
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
model = nn.LeakyReLU(0.4)
result = model(x)
lrn = nn.LocalResponseNorm(2)
signal_2d = torch.randn(32, 5, 24, 24)
result = lrn(signal_2d)
lrn = nn.LocalResponseNorm(2)
signal_4d = torch.randn(16, 5, 7, 7, 7)
result = lrn(signal_4d)
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
model = nn.LogSigmoid()
result = model(x)
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
result = nn.LogSigmoid()(x)
x = torch.tensor([[[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]])
model = nn.MaxPool1d(2)
result = model(x)
x = torch.tensor([[[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]])
model = nn.MaxPool1d(2, 1)
result = model(x)
x = torch.tensor([[[[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]]])
model = nn.MaxPool2d(2)
result = model(x)
x = torch.tensor([[[[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]]])
model = nn.MaxPool2d(2, 1)
result = model(x)
x = torch.tensor([[[[[-0.8658, 1.0869, -2.1977], [-2.1073, 1.0974, -1.4485],
    [0.588, -0.7189, 0.1089]], [[1.3036, 0.3086, -1.2245], [-0.6707, -
    0.0195, -0.1474], [0.2727, -0.4938, -0.6854]], [[0.5525, 1.0111, -
    0.1847], [0.1111, -0.6373, -0.222], [-0.5963, 0.7734, 0.0409]]]]])
model = nn.MaxPool3d(2)
result = model(x)
x = torch.tensor([[[[[-0.8658, 1.0869, -2.1977], [-2.1073, 1.0974, -1.4485],
    [0.588, -0.7189, 0.1089]], [[1.3036, 0.3086, -1.2245], [-0.6707, -
    0.0195, -0.1474], [0.2727, -0.4938, -0.6854]], [[0.5525, 1.0111, -
    0.1847], [0.1111, -0.6373, -0.222], [-0.5963, 0.7734, 0.0409]]]]])
model = nn.MaxPool3d((2, 1, 1), 1)
result = model(x)
pool = nn.MaxPool1d(2, stride=2, return_indices=True)
unpool = nn.MaxUnpool1d(2, 2)
input = torch.tensor([[[1.0, 2, 3, 4, 5, 6, 7, 8]]])
output, indices = pool(input)
result = unpool(output, indices)
pool = nn.MaxPool1d(2, stride=1, return_indices=True)
unpool = nn.MaxUnpool1d(2, stride=1)
input = torch.tensor([[[1.0, 2, 3, 4, 5, 6, 7, 8]]])
output, indices = pool(input)
result = unpool(output, indices)
pool = nn.MaxPool2d(2, stride=2, return_indices=True)
unpool = nn.MaxUnpool2d(2, stride=2)
input = torch.tensor([[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 
    10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]]])
output, indices = pool(input)
result = unpool(output, indices)
pool = nn.MaxPool2d(2, stride=2, return_indices=True)
unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
input = torch.tensor([[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 
    10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]]])
output, indices = pool(input)
result = unpool(output, indices)
pool = nn.MaxPool3d(3, stride=2, return_indices=True)
unpool = nn.MaxUnpool3d(3, stride=2)
output, indices = pool(torch.ones(2, 16, 51, 33, 15))
result = unpool(output, indices)
pool = nn.MaxPool3d(3, stride=2, return_indices=True)
unpool = nn.MaxUnpool3d(3, 2)
output, indices = pool(torch.ones(2, 16, 51, 33, 15))
result = unpool(output, indices)


class MLP(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = paddle.compat.nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = paddle.compat.nn.Linear(hidden_size, output_size)
        with torch.no_grad():
            self.fc1.weight.fill_(1.0)
            self.fc1.bias.fill_(0.1)
            self.fc2.weight.fill_(1.0)
            self.fc2.bias.fill_(0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


model = MLP(784, 256, 10)
pytorch_module = MLP(784, 256, 10)
inputs = torch.ones([64, 784])
result = model(inputs)


class MLP(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = paddle.compat.nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = paddle.compat.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


model = MLP(784, 256, 10)
result = model.__class__.__name__
x = torch.tensor([1.0, 2.0, 3.0])
module1 = torch.nn.Module()
module1.register_buffer('buffer', x)
module2 = torch.nn.Module()
module2.add_module('submodule', module1)
result = module2.submodule.buffer
x = torch.tensor([1.0, 2.0, 3.0])
module1 = torch.nn.Module()
module1.register_buffer('buffer', x)
module2 = torch.nn.Module()
module2.add_module(name='submodule', module=module1)
result = module2.submodule.buffer


def init_weights(m):
    pass


net = nn.Sequential(paddle.compat.nn.Linear(2, 2, bias=False), paddle.
    compat.nn.Linear(2, 2, bias=False))
net.apply(init_weights)
a = torch.tensor([0.0, 0.0])
result = net(a)


def init_weights(m):
    pass


net = nn.Sequential(paddle.compat.nn.Linear(2, 2, bias=False), paddle.
    compat.nn.Linear(2, 2, bias=False))
net.apply(fn=init_weights)
a = torch.tensor([0.0, 0.0])
result = net(a)
x = torch.tensor([1.0, 2.0, 3.0])
module1 = torch.nn.Module()
module1.register_buffer('buffer', x)
module1.bfloat16()
result = module1.buffer


class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.register_buffer('buf1', torch.tensor([1.0, 2.0, 4.0, 5.0]))
        self.register_buffer('buf2', torch.tensor([1.0, 2.0, 4.0, 5.0]))
        self.register_buffer('buf3', torch.tensor([1.0, 2.0, 4.0, 5.0]))

    def forward(self, x):
        pass


model = Model()
result = []
for buf in model.buffers():
    result.append(buf)


class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.register_buffer('buf1', torch.tensor([1.0, 2.0, 4.0, 5.0]))
        self.register_buffer('buf2', torch.tensor([1.0, 2.0, 4.0, 5.0]))
        self.register_buffer('buf3', torch.tensor([1.0, 2.0, 4.0, 5.0]))

    def forward(self, x):
        pass


model = Model()
result = []
for buf in model.buffers(True):
    result.append(buf)
l = paddle.compat.nn.Linear(2, 2, bias=False)
net = nn.Sequential(l, l)
result = torch.Tensor([0, 0])
for i, j in enumerate(net.children()):
    result = j(result)
x = torch.tensor([1.0, 2.0, 3.0])
module1 = torch.nn.Module()
module1.register_buffer('buffer', x)
module1.cpu()
result = module1.buffer
x = torch.tensor([1.0, 2.0, 3.0])
module1 = torch.nn.Module()
module1.register_buffer('buffer', x)
result = module1.cuda()
x = torch.tensor([1.0, 2.0, 3.0])
module1 = torch.nn.Module()
module1.register_buffer('buffer', x)
result = module1.cuda(device=0)
x = torch.tensor([1.0, 2.0, 3.0])
module1 = torch.nn.Module()
module1.register_buffer('buffer', x)
module1.double()
result = module1.buffer


class TheModelClass(nn.Module):

    def forward(self, x):
        return x


model = TheModelClass()
model.eval()
result = model.training


class TheModelClass(nn.Module):

    def forward(self, x):
        return x


model = TheModelClass()
state = model.state_dict()
model.eval().load_state_dict(state)
result = model.training
x = torch.tensor([1.0, 2.0, 3.0])
module1 = torch.nn.Module()
module1.register_buffer('buffer', x)
module1.float()
result = module1.buffer
x = torch.tensor([1.0, 2.0, 3.0])
module1 = torch.nn.Module()
module1.register_buffer('buffer', x)
result = module1.get_buffer('buffer')
x = torch.tensor([1.0, 2.0, 3.0])
module1 = torch.nn.Module()
module1.register_buffer('buffer', x)
result = module1.get_buffer(target='buffer')
x = torch.tensor([1.0, 2.0, 3.0])
module1 = torch.nn.Module()
module1.register_parameter('param1', torch.nn.parameter.Parameter(x))
result = module1.get_parameter('param1')
x = torch.tensor([1.0, 2.0, 3.0])
module1 = torch.nn.Module()
module1.register_parameter('param1', torch.nn.parameter.Parameter(x))
result = module1.get_parameter(target='param1')
x = torch.tensor([1.0, 2.0, 3.0])
module1 = torch.nn.Module()
module1.register_buffer('buffer', x)
module2 = torch.nn.Module()
module2.register_module('submodule', module1)
result = module2.get_submodule('submodule')
x = torch.tensor([1.0, 2.0, 3.0])
module1 = torch.nn.Module()
module1.register_buffer('buffer', x)
module2 = torch.nn.Module()
module2.register_module(name='submodule', module=module1)
result = module2.get_submodule(target='submodule')
x = torch.tensor([1.0, 2.0, 3.0])
module1 = torch.nn.Module()
module1.register_buffer('buffer', x)
module1.half()
result = module1.buffer


class TheModelClass(torch.nn.Module):

    def forward(self, x):
        return x


a = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
model = TheModelClass()
PATH = './tensor.pt'
paddle.save(obj=model.state_dict(), path=PATH)
model.load_state_dict(paddle.load(path=str(PATH)))
result = model(a)


class TheModelClass(torch.nn.Module):

    def forward(self, x):
        return x


a = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
model = TheModelClass()
PATH = './tensor.pt'
paddle.save(obj=model.state_dict(), path=PATH)
model.load_state_dict(paddle.load(path=str(PATH)), True)
result = model(a)
l = paddle.compat.nn.Linear(2, 2, bias=False)
net = nn.Sequential(l, l)
result = torch.Tensor([0, 0])
for i, j in enumerate(net.modules()):
    result = j(result)
l = paddle.compat.nn.Linear(2, 2, bias=False)
l1 = paddle.compat.nn.Linear(2, 2, bias=False)
model = nn.Sequential(OrderedDict([('wfs', l), ('wfs1', l1)]))
result = torch.Tensor([0, 0])
for name, module in model.named_children():
    result = module(result)
l = paddle.compat.nn.Linear(2, 2)
net = nn.Sequential(OrderedDict([('wfs', l), ('wfs1', l), ('wfs', l), (
    'wfs1', l)]))
z = net.named_modules(prefix='wfs', remove_duplicate=True)
name_list = []
for idx, m in enumerate(z):
    name_list.append(m[0])
result = name_list
l = paddle.compat.nn.Linear(2, 2)
net = nn.Sequential(OrderedDict([('wfs', l), ('wfs1', l)]))
z = net.named_modules(prefix='wfs', remove_duplicate=False)
name_list = []
for idx, m in enumerate(z):
    name_list.append(m[0])
result = name_list
result = []


class TestForHook(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear_1 = paddle.compat.nn.Linear(in_features=2, out_features=2)

    def forward(self, x):
        x1 = self.linear_1(x)
        return x, x, x1


a = TestForHook()
for a, b in a.named_parameters(prefix='wfs'):
    result.append(b)
result = result[0]
result = []


class TestForHook(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear_1 = paddle.compat.nn.Linear(in_features=2, out_features=2)

    def forward(self, x):
        x1 = self.linear_1(x)
        return x, x, x1


a = TestForHook()
for a, b in a.named_parameters(prefix='wfs', recurse=True):
    result.append(b)
result = result[0]
model = nn.ReLU()
list = model.parameters()
result = []
for i in list:
    result.append(i)
model = nn.Conv2d(1, 20, 5)
list = model.parameters()
result = []
for i in list:
    result.append(i)
weight, bias = result[0], result[1]
x = torch.tensor([1.0, 2.0, 3.0])
module = torch.nn.Module()
module.register_buffer('buffer', x)
result = module.buffer
x = torch.tensor([1.0, 2.0, 3.0])
module = torch.nn.Module()
module.register_buffer(name='buffer', tensor=x, persistent=True)
result = module.buffer
x = torch.tensor([1.0, 2.0, 3.0])
module1 = torch.nn.Module()
module1.register_buffer('buffer', x)
module2 = torch.nn.Module()
module2.register_module('submodule', module1)
result = module2.submodule.buffer
x = torch.tensor([1.0, 2.0, 3.0])
module1 = torch.nn.Module()
module1.register_buffer('buffer', x)
module2 = torch.nn.Module()
module2.register_module(name='submodule', module=module1)
result = module2.submodule.buffer
x = torch.tensor([1.0, 2.0, 3.0])
module1 = torch.nn.Module()
module1.register_parameter('param1', torch.nn.parameter.Parameter(x))
result = module1.param1
x = torch.tensor([1.0, 2.0, 3.0])
module1 = torch.nn.Module()
module1.register_parameter(name='param1', param=torch.nn.parameter.Parameter(x)
    )
result = module1.param1
x = torch.tensor([1.0, 2.0, 3.0])
module1 = torch.nn.Module()
module1.register_buffer('buffer', x)
module1.requires_grad_(True)
result = module1.buffer
x = torch.tensor([1.0, 2.0, 3.0])
module1 = torch.nn.Module()
module1.register_buffer('buffer', x)
module1.requires_grad_(requires_grad=True)
result = module1.buffer


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = paddle.compat.nn.Linear(2, 2)
        self.fc2 = paddle.compat.nn.Linear(2, 2)

    def forward(self, x):
        self.x = paddle.nn.functional.relu(x=self.fc1(x))
        x = self.x.absolute()
        x = self.fc2(x)
        return x


model = Net()
state_dict = model.state_dict()
result = []
for key, value in state_dict.items():
    result.append(key)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = paddle.compat.nn.Linear(2, 2)
        self.fc2 = paddle.compat.nn.Linear(2, 2)

    def forward(self, x):
        x = paddle.nn.functional.relu(x=self.fc1(x))
        x = self.fc2(x)
        return x


model = Net()
state_dict = model.state_dict(prefix='wfs')
result = []
for key, value in state_dict.items():
    result.append(key)
x = torch.tensor([1.0, 2.0, 3.0])
module1 = torch.nn.Module()
module1.register_buffer('buffer', x)
module1.to(dtype=torch.float32)
result = module1.buffer
x = torch.tensor([1.0, 2.0, 3.0])
module1 = torch.nn.Module()
module1.register_buffer('buffer', x)
module1.to(device='cpu')
result = module1.buffer


class TheModelClass(nn.Module):

    def forward(self, x):
        return x


model = TheModelClass()
model.train()
result = model.training


class TheModelClass(nn.Module):

    def forward(self, x):
        return x


model = TheModelClass()
state = model.state_dict()
model.train().load_state_dict(state)
result = model.training
x = torch.tensor([1.0, 2.0, 3.0])
module1 = torch.nn.Module()
module1.register_buffer('buffer', x)
module1.xpu()
result = None


class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.sum(x)
        return x


model = Model()
data_input = torch.randn(64, 1, 28, 28)
data_output = model(data_input)
data_output.backward()
model.zero_grad()
result = model.conv1.weight.grad


class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.sum(x)
        return x


model = Model()
data_input = torch.randn(64, 1, 28, 28)
data_output = model(data_input)
data_output.backward()
model.zero_grad(True)
result = model.conv1.weight.grad
choices = nn.ModuleDict({'relu': nn.ReLU()})
i = torch.tensor([1.0, 2.0])
result = choices['relu'](i)
choices = nn.ModuleDict(modules={'relu': nn.ReLU()})
i = torch.tensor([1.0, 2.0])
result = choices['relu'](i)
choices = nn.ModuleList([nn.ReLU()])
i = torch.tensor([1.0, 2.0])
result = choices[0](i)
choices = nn.ModuleList(modules=[nn.ReLU()])
i = torch.tensor([1.0, 2.0])
result = choices[0](i)
x = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
y = torch.tensor([[8.0, 3.0, 3.0], [1.4, 3.6, 0.8]])
model = nn.PairwiseDistance()
result = model(x, y)
x = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
y = torch.tensor([[8.0, 3.0, 3.0], [1.4, 3.6, 0.8]])
model = nn.PairwiseDistance(3)
result = model(x, y)
x = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
result = torch.nn.Parameter(x)
x = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
result = torch.nn.Parameter(x, requires_grad=False)
x = torch.ones(1, 9, 4, 4)
model = nn.PixelShuffle(3)
result = model(x)
x = torch.ones(1, 9, 4, 4)
model = nn.PixelShuffle(upscale_factor=3)
result = model(x)
x = torch.ones(1, 9, 12, 12)
model = nn.PixelUnshuffle(3)
result = model(x)
x = torch.ones(1, 9, 12, 12)
model = nn.PixelUnshuffle(downscale_factor=3)
result = model(x)
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
model = nn.ReLU()
result = model(x)
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
model = nn.ReLU(False)
result = model(x)
x = torch.tensor([[[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]])
model = nn.ReflectionPad1d(2)
result = model(x)
padding = model.padding
x = torch.tensor([[[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]])
model = nn.ReflectionPad1d((2, 1))
result = model(x)
padding = model.padding
x = torch.tensor([[[[-0.4106, 0.1677], [-0.6648, -0.5669]]]])
model = nn.ReflectionPad2d(1)
result = model(x)
padding = model.padding
x = torch.tensor([[[[-0.4106, 0.1677], [-0.6648, -0.5669]]]])
model = nn.ReflectionPad2d((1, 1, 1, 0))
result = model(x)
padding = model.padding
x = torch.tensor([[[[[-1.3328, -0.4948], [0.8689, 1.1423]], [[-0.2671, -
    1.0868], [1.3011, 1.0469]]]]])
model = nn.ReflectionPad3d(1)
result = model(x)
padding = model.padding
x = torch.tensor([[[[[-1.3328, -0.4948], [0.8689, 1.1423]], [[-0.2671, -
    1.0868], [1.3011, 1.0469]]]]])
model = nn.ReflectionPad3d((1, 1, 1, 1, 1, 1))
result = model(x)
padding = model.padding
x = torch.tensor([[[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]])
model = nn.ReplicationPad1d(2)
result = model(x)
x = torch.tensor([[[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]])
model = nn.ReplicationPad1d((2, 1))
result = model(x)
x = torch.tensor([[[[-0.4106, 0.1677], [-0.6648, -0.5669]]]])
model = nn.ReplicationPad2d(1)
result = model(x)
x = torch.tensor([[[[-0.4106, 0.1677], [-0.6648, -0.5669]]]])
model = nn.ReplicationPad2d((1, 1, 1, 0))
result = model(x)
x = torch.tensor([[[[[-1.3328, -0.4948], [0.8689, 1.1423]], [[-0.2671, -
    1.0868], [1.3011, 1.0469]]]]])
model = nn.ReplicationPad3d(1)
result = model(x)
x = torch.tensor([[[[[-1.3328, -0.4948], [0.8689, 1.1423]], [[-0.2671, -
    1.0868], [1.3011, 1.0469]]]]])
model = nn.ReplicationPad3d((1, 1, 1, 1, 1, 1))
result = model(x)
m = torch.nn.Sequential(torch.nn.ReLU())
result = m(torch.tensor([-1.0, 2.0, 3.0, 4.0]))
m = torch.nn.Sequential(*[torch.nn.ReLU()])
result = m(torch.tensor([-1.0, 2.0, 3.0, 4.0]))
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
model = nn.SiLU()
result = model(x)
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
model = nn.SiLU(False)
result = model(x)
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
model = nn.Sigmoid()
result = model(x)
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
result = nn.Sigmoid()(x)
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
model = nn.Softplus()
result = model(x)
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
model = nn.Softplus(2, 20)
result = model(x)
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
model = nn.Softshrink()
result = model(x)
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
model = nn.Softshrink(0.7)
result = model(x)
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
model = nn.Softsign()
result = model(x)
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
result = nn.Softsign()(x)
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
model = nn.Tanh()
result = model(x)
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
result = nn.Tanh()(x)
x = torch.tensor([-0.4, -0.2, 0.1, 0.3])
model = nn.Tanhshrink()
result = model(x)
x = torch.tensor([-0.4, -0.2, 0.1, 0.3])
result = nn.Tanhshrink()(x)
decoder_layer = paddle.nn.TransformerDecoderLayer(d_model=512, nhead=8,
    dim_feedforward=2048)
transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
memory = torch.rand(20, 32, 512)
tgt = torch.rand(20, 32, 512)
result = transformer_decoder(tgt, memory)
decoder_layer = paddle.nn.TransformerDecoderLayer(d_model=512, nhead=8,
    dim_feedforward=2048)
transformer_decoder = nn.TransformerDecoder(decoder_layer, 6)
memory = torch.rand(20, 32, 512)
tgt = torch.rand(20, 32, 512)
result = transformer_decoder(tgt, memory)
x = torch.tensor([[1.0, 5, 3], [0, 3, 2], [1, 4, 1]])
positive = torch.tensor([[5.0, 1, 2], [3, 2, 1], [3, -1, 1]])
negative = torch.tensor([[2.0, 1, -3], [1, 1, -1], [4, -2, 1]])
model = nn.TripletMarginWithDistanceLoss()
result = model(x, positive, negative)
x = torch.tensor([[1.0, 5, 3], [0, 3, 2], [1, 4, 1]])
positive = torch.tensor([[5.0, 1, 2], [3, 2, 1], [3, -1, 1]])
negative = torch.tensor([[2.0, 1, -3], [1, 1, -1], [4, -2, 1]])
model = nn.TripletMarginWithDistanceLoss(margin=2)
result = model(x, positive, negative)
input = torch.tensor([[[[1.1524, 0.4714, 0.2857], [-1.2533, -0.9829, -
    1.0981], [0.1507, -1.1431, -2.0361]], [[0.1024, -0.4482, 0.4137], [
    0.9385, 0.4565, 0.7702], [0.4135, -0.2587, 0.0482]]]])
m = torch.nn.Upsample(scale_factor=2, mode='nearest')
result = m(input)
input = torch.tensor([[[[1.1524, 0.4714, 0.2857], [-1.2533, -0.9829, -
    1.0981], [0.1507, -1.1431, -2.0361]], [[0.1024, -0.4482, 0.4137], [
    0.9385, 0.4565, 0.7702], [0.4135, -0.2587, 0.0482]]]])
m = torch.nn.Upsample(scale_factor=2, mode='bilinear')
result = m(input)
x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
model = nn.UpsamplingBilinear2d(scale_factor=2)
result = model(x)
x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
model = nn.UpsamplingBilinear2d(size=4)
result = model(x)
x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
model = nn.UpsamplingNearest2d(scale_factor=2)
result = model(x)
x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
model = nn.UpsamplingNearest2d(size=4)
result = model(x)
x = torch.tensor([[[-0.4106, 0.1677], [-0.6648, -0.5669]]])
model = nn.ZeroPad1d(1)
result = model(x)
x = torch.tensor([[[-0.4106, 0.1677], [-0.6648, -0.5669]]])
model = nn.ZeroPad1d((1, 1, 1, 0))
result = model(x)
x = torch.tensor([[[[-0.4106, 0.1677], [-0.6648, -0.5669]]]])
model = nn.ZeroPad2d(1)
result = model(x)
x = torch.tensor([[[[-0.4106, 0.1677], [-0.6648, -0.5669]]]])
model = nn.ZeroPad2d((1, 1, 1, 0))
result = model(x)
x = torch.tensor([[[[[-0.4106, 0.1677], [-0.6648, -0.5669]]]]])
model = nn.ZeroPad3d(1)
result = model(x)
x = torch.tensor([[[[[-0.4106, 0.1677], [-0.6648, -0.5669]]]]])
model = nn.ZeroPad3d((1, 1, 1, 0))
result = model(x)
modified_backend_state = [torch.nn.attention.SDPBackend.MATH]
np.random.seed(100)
x_data = np.random.randn(2, 2, 2, 2)
x = torch.tensor(x_data, dtype=torch.float32)
with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
    result = paddle.compat.nn.functional.scaled_dot_product_attention(x, x, x
        ).to(torch.float32)
    current_backends = torch.nn.attention._cur_sdpa_kernel_backends()
    assert current_backends == modified_backend_state
original_backend_state = set(torch.nn.attention._cur_sdpa_kernel_backends())
modified_backend_state = [torch.nn.attention.SDPBackend.MATH]
np.random.seed(100)
x_data = np.random.randn(2, 2, 2, 2)
x = torch.tensor(x_data, dtype=torch.float32)
current_backends = set(torch.nn.attention._cur_sdpa_kernel_backends())
assert current_backends == original_backend_state, f'Expected {original_backend_state}, got {current_backends}'
with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
    output1 = paddle.compat.nn.functional.scaled_dot_product_attention(x, x, x
        ).to(torch.float32)
    current_backends = torch.nn.attention._cur_sdpa_kernel_backends()
    assert current_backends == modified_backend_state, f'Expected {modified_backend_state}, got {current_backends}'
    output2 = paddle.compat.nn.functional.scaled_dot_product_attention(x, x, x
        ).to(torch.float32)
    current_backends = torch.nn.attention._cur_sdpa_kernel_backends()
    assert current_backends == modified_backend_state, f'Expected {modified_backend_state}, got {current_backends}'
    output3 = paddle.compat.nn.functional.scaled_dot_product_attention(x, x, x
        ).to(torch.float32)
    current_backends = torch.nn.attention._cur_sdpa_kernel_backends()
    assert current_backends == modified_backend_state, f'Expected {modified_backend_state}, got {current_backends}'
current_backends = set(torch.nn.attention._cur_sdpa_kernel_backends())
assert current_backends == original_backend_state, f'Expected {original_backend_state}, got {current_backends}'
result = output1 + output2 + output3
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
result = F.adaptive_avg_pool1d(x, 5)
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
result = F.adaptive_avg_pool1d(input=x, output_size=5)
x = torch.tensor([[[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124,
    -1.187, -1.8816], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124,
    -1.187, -1.8816]]]])
result = F.adaptive_avg_pool2d(x, 4)
x = torch.tensor([[[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124,
    -1.187, -1.8816], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124,
    -1.187, -1.8816]]]])
result = F.adaptive_avg_pool2d(x, output_size=4)
x = torch.tensor([[[[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124,
    -1.187, -1.8816], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124,
    -1.187, -1.8816]]]]])
result = F.adaptive_avg_pool3d(x, 4)
x = torch.tensor([[[[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124,
    -1.187, -1.8816], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124,
    -1.187, -1.8816]]]]])
result = F.adaptive_avg_pool3d(x, output_size=4)
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
result = nn.functional.adaptive_max_pool1d(x, 5)
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
result = nn.functional.adaptive_max_pool1d(x, output_size=5)
x = torch.tensor([[[[0.9785, 1.2013, 2.4873, -1.1891], [-0.0832, -0.5456, -
    0.5009, 1.5103], [-1.286, 1.0287, -1.3902, 0.4627], [-0.0502, -1.3924, 
    -0.3327, 0.1678]]]])
result = nn.functional.adaptive_max_pool2d(x, 5)
x = torch.tensor([[[[0.9785, 1.2013, 2.4873, -1.1891], [-0.0832, -0.5456, -
    0.5009, 1.5103], [-1.286, 1.0287, -1.3902, 0.4627], [-0.0502, -1.3924, 
    -0.3327, 0.1678]]]])
result = nn.functional.adaptive_max_pool2d(x, output_size=2)
x = torch.tensor([[[[[-1.1494, -1.3829], [0.4995, -1.3094]], [[1.0015, 
    1.4919], [-1.5187, 0.0235]]]]])
result = nn.functional.adaptive_max_pool3d(x, (1, 1, 1))
x = torch.tensor([[[[[-1.1494, -1.3829], [0.4995, -1.3094]], [[1.0015, 
    1.4919], [-1.5187, 0.0235]]]]])
result = nn.functional.adaptive_max_pool3d(x, output_size=1)
x = torch.randn(33, 16, 30)
weight = torch.randn(20, 16, 5)
result = F.conv1d(x, weight)
x = torch.randn(33, 16, 30)
weight = torch.randn(20, 16, 5)
bias = torch.randn(20)
result = F.conv1d(x, weight, bias)
x = torch.randn(33, 16, 30, 30)
weight = torch.randn(20, 16, 5, 5)
result = F.conv2d(x, weight)
x = torch.randn(33, 16, 30, 30)
weight = torch.randn(20, 16, 5, 5)
bias = torch.randn(20)
result = F.conv2d(x, weight, bias)
x = torch.randn(33, 16, 30, 30, 30)
weight = torch.randn(20, 16, 5, 5, 5)
result = F.conv3d(x, weight)
x = torch.randn(33, 16, 10, 10, 10)
weight = torch.randn(20, 16, 2, 2, 2)
bias = torch.randn(20)
result = F.conv3d(x, weight, bias)
x = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
y = torch.tensor([[8.0, 3.0, 3.0], [2.0, 3.0, 4.0]])
result = F.cosine_similarity(x, y)
x = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
y = torch.tensor([[8.0, 3.0, 3.0], [2.0, 3.0, 4.0]])
result = F.cosine_similarity(x, y, 1)
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
result = F.dropout(x)
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
result = F.dropout(x, 0.5)
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
result = F.dropout1d(x)
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
result = F.dropout1d(x, 0.5)
embedding_matrix = torch.Tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 
    2.0, 2.0], [3.0, 3.0, 3.0]])
x = torch.tensor(np.array([[0, 1], [2, 3]]))
result = torch.nn.functional.embedding(x, embedding_matrix)
embedding_matrix = torch.Tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 
    2.0, 2.0], [3.0, 3.0, 3.0]])
x = torch.tensor(np.array([[0, 1], [2, 3]]))
result = torch.nn.functional.embedding(x, embedding_matrix, padding_idx=0)
x = torch.randn(1, 3 * 2 * 2, 12)
result = F.fold(x, output_size=(4, 5), kernel_size=(2, 2))
x = torch.randn(1, 3 * 2 * 2, 12)
result = F.fold(x, output_size=(4, 5), kernel_size=(2, 2), stride=1)
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
result = F.gelu(x)
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
result = F.gelu(x, approximate='tanh')
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
result = F.glu(x)
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
result = F.glu(x, -1)
x = torch.tensor([[[[-0.6, 0.8, -0.5], [-0.5, 0.2, 1.2], [1.4, 0.3, -0.2]]]])
grid = torch.tensor([[[[0.2, 0.3], [-0.4, -0.3], [-0.9, 0.3], [-0.9, -0.6]],
    [[0.4, 0.1], [0.9, -0.8], [0.4, 0.5], [0.5, -0.2]], [[0.1, -0.8], [-0.3,
    -1.0], [0.7, 0.4], [0.2, 0.8]]]])
result = F.grid_sample(x, grid)
x = torch.tensor([[[[-0.6, 0.8, -0.5], [-0.5, 0.2, 1.2], [1.4, 0.3, -0.2]]]])
grid = torch.tensor([[[[0.2, 0.3], [-0.4, -0.3], [-0.9, 0.3], [-0.9, -0.6]],
    [[0.4, 0.1], [0.9, -0.8], [0.4, 0.5], [0.5, -0.2]], [[0.1, -0.8], [-0.3,
    -1.0], [0.7, 0.4], [0.2, 0.8]]]])
result = F.grid_sample(x, grid, padding_mode='border')
x = torch.tensor([[[[-1.2392, -0.131, -0.6679, 0.5476], [1.1738, -1.7384, -
    0.7733, 0.3261], [-0.0926, -1.0448, -1.2557, -1.5503], [0.6402, 0.9072,
    0.678, -1.9885]], [[0.0639, -1.1592, 1.4242, -0.4641], [-0.192, 0.1826,
    1.9217, -0.4359], [1.1926, -0.0247, 0.4744, -1.0216], [-0.036, -1.1656,
    0.3661, -1.8147]]]])
result = F.group_norm(x, 2)
x = torch.tensor([[[[-1.2392, -0.131, -0.6679, 0.5476], [1.1738, -1.7384, -
    0.7733, 0.3261], [-0.0926, -1.0448, -1.2557, -1.5503], [0.6402, 0.9072,
    0.678, -1.9885]], [[0.0639, -1.1592, 1.4242, -0.4641], [-0.192, 0.1826,
    1.9217, -0.4359], [1.1926, -0.0247, 0.4744, -1.0216], [-0.036, -1.1656,
    0.3661, -1.8147]]]])
result = F.group_norm(x, 2)
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
result = F.hardshrink(x)
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
result = F.hardshrink(x, 0.8)
x = torch.tensor([[[[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]]])
result = F.interpolate(x, scale_factor=2)
x = torch.tensor([[[[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]]])
result = F.interpolate(x, size=(2, 3))
input = torch.tensor([[[1.1524, 0.4714, 0.2857], [-1.2533, -0.9829, -1.0981
    ], [0.1507, -1.1431, -2.0361]], [[0.1024, -0.4482, 0.4137], [0.9385, 
    0.4565, 0.7702], [0.4135, -0.2587, 0.0482]]])
data = torch.tensor([1.0, 1.0, 1.0])
result = F.layer_norm(input, [3], data, data)
input = torch.tensor([[[1.1524, 0.4714, 0.2857], [-1.2533, -0.9829, -1.0981
    ], [0.1507, -1.1431, -2.0361]], [[0.1024, -0.4482, 0.4137], [0.9385, 
    0.4565, 0.7702], [0.4135, -0.2587, 0.0482]]])
data = torch.tensor([1.0, 1.0, 1.0])
result = F.layer_norm(input=input, normalized_shape=[3], weight=data, bias=data
    )
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
result = F.leaky_relu(x)
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
result = F.leaky_relu(x, 0.06)
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
F.leaky_relu_(input=x, negative_slope=0.08)
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
F.leaky_relu_(x, 0.06)
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
result = F.logsigmoid(x)
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
result = F.logsigmoid(input=x)
input = torch.tensor([[[-0.5743, 0.4889, -0.0878, 0.421, -0.0844], [0.3614,
    0.8458, -0.6152, 0.6894, 0.2927], [-0.0087, 0.1098, 0.1783, -0.6953, 
    0.5519], [0.3789, -0.056, -0.409, -0.107, -1.0139], [0.9204, 1.0817, -
    2.6126, 0.4244, 0.3272]]])
result = torch.nn.functional.lp_pool1d(input, 1, 2)
input = torch.tensor([[[0.643, 0.4511, -1.6757, 1.7116], [-0.2288, -0.4111,
    -1.3602, 0.2685], [0.2363, 1.9341, 0.8522, -0.1846], [1.6496, -0.0675, 
    -0.7208, -1.0018]], [[-0.3183, 0.8029, -0.4993, 1.0598], [-0.4952, -
    0.9536, 0.1954, 0.0551], [1.2257, 0.7517, 0.4063, -1.2151], [-1.3562, 
    0.3547, 1.1147, 1.2898]], [[0.1205, -0.1889, 0.5086, -0.808], [0.3156, 
    -0.8298, 2.0242, -0.9184], [-0.4005, 1.3586, 0.6205, -0.7487], [1.6239,
    0.29, 0.9671, 1.2961]], [[-1.1996, -0.2201, -0.9466, -0.7264], [-0.0313,
    0.8284, -0.3588, 1.3522], [-0.0991, -0.5112, -0.1785, 2.0903], [-1.3286,
    -0.9333, -0.1404, 1.2582]]])
result = torch.nn.functional.lp_pool1d(input, 4, 2, 2)
input = torch.tensor([[[[-0.5743, 0.4889, -0.0878, 0.421, -0.0844], [0.3614,
    0.8458, -0.6152, 0.6894, 0.2927], [-0.0087, 0.1098, 0.1783, -0.6953, 
    0.5519], [0.3789, -0.056, -0.409, -0.107, -1.0139], [0.9204, 1.0817, -
    2.6126, 0.4244, 0.3272]]]])
result = torch.nn.functional.lp_pool2d(input, 1, 3, stride=2)
input = torch.tensor([[[[0.643, 0.4511, -1.6757, 1.7116], [-0.2288, -0.4111,
    -1.3602, 0.2685], [0.2363, 1.9341, 0.8522, -0.1846], [1.6496, -0.0675, 
    -0.7208, -1.0018]], [[-0.3183, 0.8029, -0.4993, 1.0598], [-0.4952, -
    0.9536, 0.1954, 0.0551], [1.2257, 0.7517, 0.4063, -1.2151], [-1.3562, 
    0.3547, 1.1147, 1.2898]], [[0.1205, -0.1889, 0.5086, -0.808], [0.3156, 
    -0.8298, 2.0242, -0.9184], [-0.4005, 1.3586, 0.6205, -0.7487], [1.6239,
    0.29, 0.9671, 1.2961]], [[-1.1996, -0.2201, -0.9466, -0.7264], [-0.0313,
    0.8284, -0.3588, 1.3522], [-0.0991, -0.5112, -0.1785, 2.0903], [-1.3286,
    -0.9333, -0.1404, 1.2582]]]])
result = torch.nn.functional.lp_pool2d(input, 2, 4, stride=2)
input = torch.tensor([[[1.1524, 0.4714, 0.2857], [-1.2533, -0.9829, -1.0981
    ], [0.1507, -1.1431, -2.0361]]])
result = F.max_pool1d(input, 3)
input = torch.tensor([[[1.1524, 0.4714, 0.2857], [-1.2533, -0.9829, -1.0981
    ], [0.1507, -1.1431, -2.0361]]])
result = F.max_pool1d(input, 3, stride=2)
input = torch.tensor([[[[1.1524, 0.4714, 0.2857], [-1.2533, -0.9829, -
    1.0981], [0.1507, -1.1431, -2.0361]], [[0.1024, -0.4482, 0.4137], [
    0.9385, 0.4565, 0.7702], [0.4135, -0.2587, 0.0482]]]])
result = F.max_pool2d(input, 3)
input = torch.tensor([[[[1.1524, 0.4714, 0.2857], [-1.2533, -0.9829, -
    1.0981], [0.1507, -1.1431, -2.0361]], [[0.1024, -0.4482, 0.4137], [
    0.9385, 0.4565, 0.7702], [0.4135, -0.2587, 0.0482]]]])
result = F.max_pool2d(input, (3, 1))
input = torch.arange(4800, dtype=torch.float32).reshape(2, 3, 8, 10, 10)
result = F.max_pool3d(input, 3)
input = torch.arange(4800, dtype=torch.float32).reshape(2, 3, 8, 10, 10)
result, indices = F.max_pool3d(input, 3, 1, 1, 2, True, True)
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
result = F.mish(x)
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
result = F.mish(input=x)
x = torch.tensor([[[[-1.2392, -0.131, -0.6679, 0.5476], [1.1738, -1.7384, -
    0.7733, 0.3261], [-0.0926, -1.0448, -1.2557, -1.5503], [0.6402, 0.9072,
    0.678, -1.9885]], [[0.0639, -1.1592, 1.4242, -0.4641], [-0.192, 0.1826,
    1.9217, -0.4359], [1.1926, -0.0247, 0.4744, -1.0216], [-0.036, -1.1656,
    0.3661, -1.8147]]]])
result = F.normalize(x)
x = torch.tensor([[[[-1.2392, -0.131, -0.6679, 0.5476], [1.1738, -1.7384, -
    0.7733, 0.3261], [-0.0926, -1.0448, -1.2557, -1.5503], [0.6402, 0.9072,
    0.678, -1.9885]], [[0.0639, -1.1592, 1.4242, -0.4641], [-0.192, 0.1826,
    1.9217, -0.4359], [1.1926, -0.0247, 0.4744, -1.0216], [-0.036, -1.1656,
    0.3661, -1.8147]]]])
result = F.normalize(x, 3.0, 1)
x = torch.tensor([1, 2, 0, 3, 5]) % 3
result = F.one_hot(x)
x = torch.tensor([1, 2, 0, 3, 5]) % 3
result = F.one_hot(x, 3)
a = [1.3192, 1.9915, 1.9674, 1.7151]
b = [1.3492, 0.1915, 2.9434, 1.4151]
x1 = torch.tensor(a)
x2 = torch.tensor(b)
result = torch.nn.functional.pairwise_distance(x1, x2, 2.0, 1e-06, False)
a = [1.3192, 1.9915, 1.9674, 1.7151]
b = [1.3492, 0.1915, 2.9434, 1.4151]
x1 = torch.tensor(a)
x2 = torch.tensor(b)
result = torch.nn.functional.pairwise_distance(x1=x1, x2=x2, p=1.0, eps=
    1e-06, keepdim=False)
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
result = F.relu(x)
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
result = F.relu(x, False)
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
result = F.relu_(x)
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
result = F.relu_(input=x)
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
result = F.sigmoid(x)
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
result = F.sigmoid(input=x)
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
result = F.silu(x)
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
result = F.silu(x, True)
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
result = F.softplus(x)
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
result = F.softplus(x, 3, 15)
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
result = F.softshrink(x)
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
result = F.softshrink(x, 0.3)
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
result = F.tanh(x)
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
result = F.tanh(input=x)
x = torch.randn(5, 5)
result = torch.nn.init._calculate_fan_in_and_fan_out(x)
x = torch.randn(5, 5, 5)
result = torch.nn.init._calculate_fan_in_and_fan_out(x)
conv = torch.nn.Conv2d(4, 6, (3, 3))
torch.nn.init.constant_(conv.weight, val=0.2)
result = conv.weight
conv = torch.nn.Conv2d(3, 6, (3, 3))
torch.nn.init.constant_(conv.weight, val=2)
result = conv.weight
conv = torch.nn.Conv2d(4, 6, (3, 3))
torch.nn.init.dirac_(conv.weight)
result = conv.weight
conv = torch.nn.Conv2d(3, 6, (3, 3))
torch.nn.init.dirac_(conv.weight, 1)
result = conv.weight
conv = torch.empty(3, 5)
torch.nn.init.eye_(conv)
result = conv
conv = torch.empty(3, 5)
torch.nn.init.eye_(tensor=conv)
result = conv
conv = torch.nn.Conv2d(4, 6, (3, 3))
torch.nn.init.normal_(conv.weight)
result = conv.weight
conv = torch.nn.Conv2d(4, 6, (3, 3))
torch.nn.init.normal_(conv.weight, 0.2, 2.0)
result = conv.weight
conv = torch.nn.Conv2d(4, 6, (3, 3))
torch.nn.init.ones_(conv.weight)
result = conv.weight
conv = torch.nn.Conv2d(3, 6, (3, 3))
torch.nn.init.ones_(tensor=conv.weight)
result = conv.weight
conv = torch.nn.Conv2d(4, 6, (3, 3))
torch.nn.init.orthogonal_(conv.weight)
result = conv.weight
conv = torch.nn.Conv2d(3, 6, (3, 3))
torch.nn.init.orthogonal_(tensor=conv.weight)
result = conv.weight
conv = torch.nn.Conv2d(4, 6, (3, 3))
torch.nn.init.trunc_normal_(conv.weight)
result = conv.weight
conv = torch.nn.Conv2d(3, 6, (3, 3))
torch.nn.init.trunc_normal_(tensor=conv.weight, mean=1.0, std=2.0, a=-1.0,
    b=1.0)
result = conv.weight
conv = torch.nn.Conv2d(4, 6, (3, 3))
torch.nn.init.uniform_(conv.weight)
result = conv.weight
conv = torch.nn.Conv2d(3, 6, (3, 3))
torch.nn.init.uniform_(tensor=conv.weight)
result = conv.weight
conv = torch.nn.Conv2d(4, 6, (3, 3))
torch.nn.init.xavier_normal_(conv.weight)
result = conv.weight
conv = torch.nn.Conv2d(3, 6, (3, 3))
torch.nn.init.xavier_normal_(tensor=conv.weight)
result = conv.weight
conv = torch.nn.Conv2d(4, 6, (3, 3))
torch.nn.init.xavier_uniform_(conv.weight)
result = conv.weight
conv = torch.nn.Conv2d(3, 6, (3, 3))
torch.nn.init.xavier_uniform_(tensor=conv.weight)
result = conv.weight
conv = torch.nn.Conv2d(4, 6, (3, 3))
torch.nn.init.zeros_(conv.weight)
result = conv.weight
conv = torch.nn.Conv2d(3, 6, (3, 3))
torch.nn.init.zeros_(tensor=conv.weight)
result = conv.weight
x = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
result = torch.nn.parameter.Parameter(x)
x = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
result = torch.nn.parameter.Parameter(x, requires_grad=False)
model = paddle.compat.nn.Linear(10, 20)
result = nn.utils.parameters_to_vector(model.parameters())
model = paddle.compat.nn.Linear(10, 20)
result = nn.utils.parameters_to_vector(parameters=model.parameters())
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0])
c = torch.tensor([6.0])
result = pad_sequence([a, b, c])
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0])
c = torch.tensor([6.0])
result = pad_sequence([a, b, c], batch_first=True)
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0])
c = torch.tensor([6.0])
padded = pad_sequence([a, b, c], batch_first=True)
lengths = torch.tensor([3, 2, 1])
result = unpad_sequence(padded, lengths, batch_first=True)
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0])
padded = pad_sequence([a, b], batch_first=False)
lengths = torch.tensor([3, 2])
result = unpad_sequence(padded, lengths, batch_first=False)
model = paddle.compat.nn.Linear(10, 20)
a = nn.utils.parameters_to_vector(model.parameters())
b = nn.utils.vector_to_parameters(a, model.parameters())
result = a.detach()
model = paddle.compat.nn.Linear(10, 20)
a = nn.utils.parameters_to_vector(model.parameters())
b = nn.utils.vector_to_parameters(vec=a, parameters=model.parameters())
result = a.detach()
x = torch.tensor([1.0], requires_grad=True)
with torch.no_grad():
    y = x * 2


@torch.no_grad()
def doubler(x):
    return x * 2


x = torch.tensor([1.0], requires_grad=True)
y = doubler(x)
result = torch.nonzero(torch.tensor([1, 1, 1, 0, 1]))
result = torch.nonzero(torch.tensor([[0.6, 0.0, 0.0, 0.0], [0.0, 0.4, 0.0, 
    0.0], [0.0, 0.0, 1.2, 0.0], [0.0, 0.0, 0.0, -0.4]]))
input = torch.tensor([[[-12.0, -11.0, -10.0, -9.0], [-8.0, -7.0, -6.0, -5.0
    ], [-4.0, -3.0, -2.0, -1.0]], [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 
    7.0], [8.0, 9.0, 10.0, 11.0]]])
result = torch.norm(input, p='fro')
input = torch.tensor([[-12.0, -11.0, -10.0, -9.0], [-8.0, -7.0, -6.0, -5.0],
    [-4.0, -3.0, -2.0, -1.0]])
result = torch.norm(input, p='nuc')
result = torch.normal(torch.arange(1.0, 11.0), torch.arange(1, 11))
result = torch.normal(mean=0.5, std=torch.arange(1.0, 6.0))
result = torch.not_equal(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 
    1], [4, 4]]))
input = torch.tensor([[1, 2], [3, 4]])
other = torch.tensor([[1, 1], [4, 4]])
result = torch.not_equal(input, other)
result = torch.ones(3)
result = torch.ones(3, 5)
input = torch.empty(2, 3)
result = torch.ones_like(input)
result = torch.ones_like(torch.empty(2, 3))
x = torch.tensor([1.0, 2, 3])
y = torch.tensor([1.0, 2, 3, 4])
result = torch.outer(x, y)
x = torch.tensor([1.0, 2, 3])
y = torch.tensor([1.0, 2, 3, 4])
result = torch.outer(input=x, vec2=y)
x = torch.tensor([[1, 2, 3], [3, 4, 6]])
result = torch.permute(x, (1, 0))
x = torch.tensor([[1, 2, 3], [3, 4, 6]])
result = torch.permute(x, [1, 0])
result = torch.pi
abs = torch.tensor([1, 2], dtype=torch.float64)
angle = torch.tensor([np.pi / 2, 5 * np.pi / 4], dtype=torch.float64)
result = torch.polar(abs, angle)
abs = torch.tensor([1, 2], dtype=torch.float64)
angle = torch.tensor([np.pi / 2, 5 * np.pi / 4], dtype=torch.float64)
out = torch.tensor([1, 2], dtype=torch.complex128)
result = torch.polar(abs, angle, out=out)
a = torch.tensor([0.4331, 1.2475, 0.6834, -0.2791])
result = torch.pow(a, 2)
a = torch.tensor([0.4331, 1.2475, 0.6834, -0.2791])
b = torch.tensor([1, 2, 3, 4])
result = torch.pow(a, b)
input = torch.tensor([1.4907, 1.0593, 1.5696])
result = torch.prod(input)
input = torch.tensor([[1.4907, 1.0593, 1.5696], [1.4907, 1.0593, 1.5696]])
result = torch.prod(input, 1)
>>>>>>result = torch.quantile(torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.
    float64), 0.6)
>>>>>>result = torch.quantile(torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.
    float64), 0.6, dim=None)
result = torch.rand(3)
result = torch.rand(3, 5)
a = torch.zeros(3, 4, dtype=torch.float64)
result = torch.rand_like(a)
a = torch.zeros(3, 4, dtype=torch.float64)
result = torch.rand_like(a, dtype=torch.float32, requires_grad=True)
result = torch.randn(3)
result = torch.randn(3, 5)
a = torch.zeros(3, 4, dtype=torch.float64)
result = torch.randn_like(a)
a = torch.zeros(3, 4, dtype=torch.float64)
result = torch.randn_like(a, dtype=torch.float32, requires_grad=True)
torch.manual_seed(100)
result = torch.random.initial_seed()
result = torch.randperm(5)
n = 5
result = torch.randperm(n)
result = torch.range(1, 4)
result = torch.range(1, 4, 0.5)
a = torch.tensor([[4, 9], [23, 2]])
result = torch.ravel(a)
result = torch.ravel(torch.tensor([[4, 9], [23, 2]]))
result = torch.reciprocal(torch.tensor([-0.4595, -2.1219, -1.4314, 0.7298]))
a = torch.tensor([-0.4595, -2.1219, -1.4314, 0.7298])
result = torch.reciprocal(a)
a = torch.tensor([-3.0, -2, -1, 1, 2, 3])
result = torch.remainder(a, 2.0)
a = torch.tensor([1, 2, 3, 4, 5])
result = torch.remainder(a, torch.tensor(1.5))
x = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
result = torch.renorm(x, 1, 0, 5)
x = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
result = torch.renorm(input=x, p=1, dim=0, maxnorm=5)
a = torch.tensor([[4, 9], [23, 2]])
result = torch.repeat_interleave(a, 3, 0)
a = torch.tensor([[4, 9], [23, 2]])
result = torch.repeat_interleave(input=a, repeats=3, dim=1)
a = torch.arange(4.0)
result = torch.reshape(a, (2, 2))
a = torch.arange(9)
shape = 3, 3
result = torch.reshape(a, shape)
x = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
result = torch.roll(x, 1)
x = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
result = torch.roll(x, 1, 0)
a = torch.tensor([[[0.9254, -0.6213], [-0.5787, 1.6843]], [[0.3242, -0.9665
    ], [0.4539, -0.0887]], [[1.1336, -0.4025], [-0.7089, 0.9032]]])
result = torch.round(a)
result = torch.round(torch.tensor([[[0.9254, -0.6213], [-0.5787, 1.6843]],
    [[0.3242, -0.9665], [0.4539, -0.0887]], [[1.1336, -0.4025], [-0.7089, 
    0.9032]]]))
result = torch.rsqrt(torch.tensor([0.297, 1.542, 4]))
a = torch.tensor([0.297, 1.542, 4])
result = torch.rsqrt(a)
input = torch.arange(15).reshape([3, 5]).astype(torch.float32)
index = torch.tensor([[0], [1], [2]])
result = torch.scatter(input, 1, index, 1.0)
input = torch.arange(15).reshape([3, 5]).astype(torch.float32)
index = torch.tensor([[0], [1], [2]])
result = torch.scatter(input=input, dim=1, index=index, value=1.0)
src = torch.ones((1, 5))
index = torch.tensor([[0, 1, 2, 0, 0]])
input = torch.zeros(3, 5, dtype=src.dtype)
result = torch.scatter_add(input, 0, index, src)
src = torch.ones((2, 5))
index = torch.tensor([[0, 1, 2, 0, 0], [0, 1, 2, 2, 2]])
input = torch.zeros(3, 5, dtype=src.dtype)
result = torch.scatter_add(input, 0, index, src)
src = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
index = torch.tensor([0, 1, 0, 1, 2, 1])
input = torch.tensor([1.0, 2.0, 3.0, 4.0])
type = 'sum'
result = torch.scatter_reduce(input, 0, index, src, reduce=type)
src = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
index = torch.tensor([0, 1, 0, 1, 2, 1])
input = torch.tensor([1.0, 2.0, 3.0, 4.0])
result = torch.scatter_reduce(input=input, dim=0, index=index, src=src,
    reduce='sum', include_self=False)
x = torch.tensor([[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]])
values = torch.tensor([[3, 6, 9], [3, 6, 9]])
result = torch.searchsorted(x, values)
x = torch.tensor([[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]])
values = torch.tensor([[3, 6, 9], [3, 6, 9]])
result = torch.searchsorted(x, values, out_int32=True)
torch.set_default_dtype(torch.float64)
result = torch.tensor([1.2, 3])
torch.set_default_dtype(torch.float64)
result = torch.tensor([1.2, 3.0j])
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
result = torch.sigmoid(x)
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
result = torch.sigmoid(input=x)
result = torch.sign(torch.tensor([0.9213, 1.0887, -0.8858, -1.7683]))
a = torch.tensor([0.9213, 1.0887, -0.8858, -1.7683])
result = torch.sign(a)
result = torch.sin(torch.tensor([1.4309, 1.2706, -0.8562, 0.9796]))
a = torch.tensor([1.4309, 1.2706, -0.8562, 0.9796])
result = torch.sin(a)
result = torch.sinh(torch.tensor([1.4309, 1.2706, -0.8562, 0.9796]))
a = torch.tensor([1.4309, 1.2706, -0.8562, 0.9796])
result = torch.sinh(a)
input = torch.tensor([[-1.2837, -0.0297, 0.0355], [0.9112, -1.7526, -0.4061]])
result = torch.softmax(input, dim=0)
input = torch.tensor([[-1.2837, -0.0297, 0.0355], [0.9112, -1.7526, -0.4061]])
result = torch.softmax(input, dim=1)
result = torch.special.expm1(torch.tensor([0.0, -2.0, 3.0]))
a = torch.tensor([-1.0, -2.0, 3.0])
result = torch.special.expm1(a)
result = torch.special.i0(torch.tensor([1.0, 2.0, 3.0]))
x = torch.tensor([1.0, 2.0, 3.0])
result = torch.special.i0(x)
x = torch.tensor([1.0, 2.0, 3.0])
result = torch.special.i0e(x)
x = torch.tensor([1.0, 2.0, 3.0])
result = torch.special.i0e(input=x)
result = torch.special.i1(torch.tensor([1.0, 2.0, 3.0]))
x = torch.tensor([1.0, 2.0, 3.0])
result = torch.special.i1(x)
result = torch.special.i1e(torch.tensor([1.0, 2.0, 3.0]))
x = torch.tensor([1.0, 2.0, 3.0])
result = torch.special.i1e(x)
input = torch.tensor([1.4907, 1.0593, 1.5696])
result = torch.special.logsumexp(input, 0)
input = torch.tensor([[1.4907, 1.0593, 1.5696], [1.4907, 1.0593, 1.5696]])
result = torch.special.logsumexp(input, 1)
x = torch.tensor([[[2.0, 3.0, 4.0, 5.0], [3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 
    8.0, 9.0]], [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [6.0, 7.0, 8.0,
    9.0]]])
result = torch.special.softmax(x, -1)
x = torch.tensor([[[2.0, 3.0, 4.0, 5.0], [3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 
    8.0, 9.0]], [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [6.0, 7.0, 8.0,
    9.0]]])
result = torch.special.softmax(x, dim=1)
result = torch.sqrt(torch.tensor([0.297, 1.542, 4]))
a = torch.tensor([0.297, 1.542, 4])
result = torch.sqrt(a)
result = torch.square(torch.tensor([0.297, 1.542, 4]))
a = torch.tensor([0.297, 1.542, 4])
result = torch.square(a)
x = torch.zeros(2, 1, 2, 1, 2)
result = torch.squeeze(x)
result = torch.squeeze(torch.zeros(2, 1, 2, 1, 2))
x = torch.zeros(2, 3)
y = torch.zeros(2, 3)
result = torch.stack((x, y))
result = torch.stack((torch.zeros(2, 3), torch.zeros(2, 3)))
input = torch.tensor([1.4907, 1.0593, 1.5696])
result = torch.std(input, unbiased=False)
input = torch.tensor([[1.4907, 1.0593, 1.5696], [1.4907, 1.0593, 1.5696]])
result = torch.std(input, unbiased=False)
a = torch.tensor([0.595, -0.0872, 2.3298, -0.2972])
b = torch.tensor([1, 1, 1, 0])
result = torch.sub(a, b)
a = torch.tensor([0.595, -0.0872, 2.3298, -0.2972])
b = torch.tensor([1, 1, 1, 0])
result = torch.sub(input=a, other=b)
input = torch.tensor([1.4907, 1.0593, 1.5696])
result = torch.sum(input)
input = torch.tensor([[1.4907, 1.0593, 1.5696], [1.4907, 1.0593, 1.5696]])
result = torch.sum(input, 1)
x = torch.tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])
result = torch.swapaxes(x, 0, 1)
x = torch.tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])
result = torch.swapaxes(input=x, axis0=0, axis1=1)
x = torch.tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])
result = torch.swapdims(x, 0, 1)
x = torch.tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])
result = torch.swapdims(input=x, dim0=0, dim1=1)
x = torch.zeros(2, 3)
result = torch.t(x)
x = torch.zeros(2)
result = torch.t(x)
x = torch.tensor([[1, 2, 3], [3, 4, 6]])
idx = torch.tensor([[0]])
result = torch.take_along_dim(x, idx, 1)
x = torch.tensor([[1, 2, 3], [3, 4, 6]])
idx = torch.tensor([[0]])
result = torch.take_along_dim(input=x, indices=idx, dim=0)
result = torch.tan(torch.tensor([1.4309, 1.2706, -0.8562, 0.9796]))
a = torch.tensor([1.4309, 1.2706, -0.8562, 0.9796])
result = torch.tan(a)
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
result = torch.tanh(x)
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
result = torch.tanh(input=x)
result = torch.tensor([2, 3])
data = [2, 3]
result = torch.tensor(data)
a = torch.arange(8)
result = torch.tensor_split(a, 3)
a = torch.arange(7)
result = torch.tensor_split(a, sections=3)
x = torch.tensor([1.0, 2.0, 3.0])
torch.testing.assert_close(x, x)
x = torch.tensor([1.0, 2.0, 3.0])
torch.testing.assert_close(actual=x, expected=x)
x = torch.tensor([1, 2, 3])
result = torch.tile(x, (2,))
x = torch.tensor([[1, 2], [0, 6]])
result = torch.tile(x, (2, 3))
x = torch.tensor([1, 2, 3, 4, 5])
result, index = torch.topk(x, 3)
x = torch.tensor([1, 2, 3, 4, 5])
res = torch.topk(x, 3)
result, index = res[0], res[1]
src = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
result = src.to(torch.torch.int32)
result = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).to(torch.torch.int32)
a = torch.Tensor([[1.0, 2.0], [3.0, 4.0]])
result = torch.transpose(a, dim0=0, dim1=1)
a = torch.Tensor([[1.0, 2.0], [3.0, 4.0]])
result = torch.transpose(a, 0, 1)
x = torch.tensor([[-1.0813, -0.8619, 0.7105], [0.0935, 0.138, 2.2112], [-
    0.3409, -0.9828, 0.0289]])
result = torch.tril(x)
x = torch.tensor([[-1.0813, -0.8619, 0.7105], [0.0935, 0.138, 2.2112], [-
    0.3409, -0.9828, 0.0289]])
result = torch.tril(x, 1)
x = torch.tensor([[-1.0813, -0.8619, 0.7105], [0.0935, 0.138, 2.2112], [-
    0.3409, -0.9828, 0.0289]])
result = torch.triu(x)
x = torch.tensor([[-1.0813, -0.8619, 0.7105], [0.0935, 0.138, 2.2112], [-
    0.3409, -0.9828, 0.0289]])
result = torch.triu(x, 1)
a = torch.tensor([4.67, 9.76, 8.53])
b = torch.tensor([3.5, 3.9, 1.83])
result = torch.true_divide(a, b)
a = torch.tensor([[4.0, 9.0, 8.0]])
b = torch.tensor([2.0, 3.0, 4.0])
result = torch.true_divide(a, b)
src = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
result = src.to(torch.uint8)
result = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).to(torch.uint8)
x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
result = torch.unbind(x)
x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
result = torch.unbind(x, 1)
a = torch.tensor([[0.218, 1.0558, 0.1608, 0.9245], [1.3794, 1.409, 0.2514, 
    -0.8818], [-0.4561, 0.5123, 1.7505, -0.4094]])
result = torch.unflatten(a, -1, (2, 2))
a = torch.tensor([[0.218, 1.0558, 0.1608, 0.9245], [1.3794, 1.409, 0.2514, 
    -0.8818], [-0.4561, 0.5123, 1.7505, -0.4094]])
result = torch.unflatten(a, 1, (2, 2))
x = torch.tensor([1, 1, 2, 2, 3, 1, 1, 2])
result = torch.unique_consecutive(x)
x = torch.tensor([1, 1, 2, 2, 3, 1, 1, 2])
result, inverse_indices = torch.unique_consecutive(x, return_inverse=True)
x = torch.zeros(2, 2, 2)
result = torch.unsqueeze(x, 0)
result = torch.unsqueeze(torch.zeros(2, 2, 1, 2), 3)
paddle.utils.cpp_extension.CppExtension(sources=['extension.cpp'],
    extra_compile_args=['-g'])
dic = {'build_ext': BuildExtension}
result = True
paddle.utils.cpp_extension.CppExtension(sources=['extension.cpp'],
    extra_compile_args=['-g'])
dic = {'build_ext': BuildExtension}
result = True
paddle.utils.cpp_extension.CppExtension(sources=['extension.cpp'],
    extra_compile_args=['-g'])
dic = {'build_ext': BuildExtension.with_options}
result = True
result = CUDA_HOME


class MyIterableDataset(torch.utils.data.IterableDataset):

    def __init__(self, start, end):
        super(MyIterableDataset).__init__()
        assert end > start, 'this example code only works with end >= start'
        self.start = start
        self.end = end

    def __iter__(self):
        iter_start = self.start
        iter_end = self.end
        return iter(range(iter_start, iter_end))


dataset = torch.utils.data.ChainDataset([MyIterableDataset(start=3, end=7),
    MyIterableDataset(start=3, end=7)])
result = []
for d in dataset:
    result.append(d)


class MyIterableDataset(data.IterableDataset):

    def __init__(self, start, end):
        super(MyIterableDataset).__init__()
        assert end > start, 'this example code only works with end >= start'
        self.start = start
        self.end = end

    def __iter__(self):
        iter_start = self.start
        iter_end = self.end
        return iter(range(iter_start, iter_end))


dataset = data.ChainDataset([MyIterableDataset(start=1, end=10),
    MyIterableDataset(start=1, end=3)])
result = []
for d in dataset:
    result.append(d)


class RandomDataset(Dataset):

    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __getitem__(self, idx):
        image = np.arange(5).astype('float32')
        label = np.array([idx]).astype('int64')
        return torch.tensor(image), torch.tensor(label)

    def __len__(self):
        return self.num_samples


dataset = ConcatDataset([RandomDataset(2), RandomDataset(2)])
result = []
for i in range(len(dataset)):
    result.append(dataset[i])


class RandomDataset(data.Dataset):

    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __getitem__(self, idx):
        image = np.arange(5).astype('float32')
        label = np.array([idx]).astype('int64')
        return torch.tensor(image), torch.tensor(label)

    def __len__(self):
        return self.num_samples


dataset = data.ConcatDataset(datasets=[RandomDataset(2), RandomDataset(2)])
result = []
for i in range(len(dataset)):
    result.append(dataset[i])


class Data(Dataset):

    def __init__(self):
        self.x = [1, 2, 3, 4]

    def __getitem__(self, idx):
        return self.x[idx]

    def __len__(self):
        return len(self.x)


data = Data()
>>>>>>result = torch.utils.data.__len__()


class Data(data.Dataset):

    def __init__(self):
        self.x = [1, 2, 3, 4]

    def __getitem__(self, idx):
        return self.x[idx]

    def __len__(self):
        return len(self.x)


my_data = Data()
result = my_data.__getitem__(0)


class MyIterableDataset(IterableDataset):

    def __init__(self, start, end):
        super(MyIterableDataset).__init__()
        assert end > start, 'this example code only works with end >= start'
        self.start = start
        self.end = end

    def __iter__(self):
        return iter(range(self.start, self.end))


ds = MyIterableDataset(start=3, end=7)
result = []
for i in ds:
    result.append(i)


class MyIterableDataset(data.IterableDataset):

    def __init__(self, start, end):
        super(MyIterableDataset).__init__()
        assert end > start, 'this example code only works with end >= start'
        self.start = start
        self.end = end

    def __iter__(self):
        return iter(range(self.start, self.end))


ds = MyIterableDataset(start=3, end=7)
result = next(ds.__iter__())


class Data(Dataset):

    def __init__(self):
        self.x = np.arange(0, 100, 1)

    def __getitem__(self, idx):
        return self.x[idx]

    def __len__(self):
        return self.x.shape[0]


class MySampler(Sampler):

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)


data = Data()
s = MySampler(data)
result = []
for d in s:
    result.append(d)


class Data(data.Dataset):

    def __init__(self):
        self.x = np.arange(0, 100, 1)

    def __getitem__(self, idx):
        return self.x[idx]

    def __len__(self):
        return self.x.shape[0]


class MySampler(data.Sampler):

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)


my_data = Data()
s = MySampler(data_source=my_data)
result = []
for d in s:
    result.append(d)


class MyDataset(Dataset):

    def __init__(self):
        self.x = np.arange(0, 100, 1)

    def __getitem__(self, idx):
        return self.x[idx]

    def __len__(self):
        return len(self.x)


s = SequentialSampler(MyDataset())
result = []
for d in s:
    result.append(d)


class MyDataset(data.Dataset):

    def __init__(self):
        self.x = np.arange(0, 100, 1)

    def __getitem__(self, idx):
        return self.x[idx]

    def __len__(self):
        return len(self.x)


my_data = MyDataset()
s = data.SequentialSampler(data_source=my_data)
result = []
for d in s:
    result.append(d)
result = torch.utils.data.get_worker_info()


class MyIterableDataset(IterableDataset):

    def __init__(self, start, end):
        super(MyIterableDataset).__init__()
        assert end > start, 'this example code only works with end >= start'
        self.start = start
        self.end = end

    def __iter__(self):
        worker_info = get_worker_info()
        iter_start = self.start
        iter_end = self.end
        return iter(range(iter_start, iter_end))


dataset = ChainDataset([MyIterableDataset(start=1, end=10),
    MyIterableDataset(start=1, end=3)])
result = []
for d in dataset:
    result.append(d)


class Data(torch.utils.data.Dataset):

    def __init__(self):
        self.x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    def __getitem__(self, idx):
        return self.x[idx]

    def __len__(self):
        return len(self.x)


data = Data()
datasets = torch.utils.data.random_split(data, [3, 7])
results = []
for d in datasets:
    results.append(d.__len__())


class Data(data.Dataset):

    def __init__(self):
        self.x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    def __getitem__(self, idx):
        return self.x[idx]

    def __len__(self):
        return len(self.x)


my_data = Data()
datasets = data.random_split(my_data, [3, 3, 4])
results = []
for d in datasets:
    results.append(d.__len__())
input = torch.tensor([1.4907, 1.0593, 1.5696])
result = torch.var(input, unbiased=False)
input = torch.tensor([[1.4907, 1.0593, 1.5696], [1.4907, 1.0593, 1.5696]])
result = torch.var(input, unbiased=False)
x = torch.tensor([[1.6116, -0.5772], [-1.4606, -0.912]])
result = torch.view_as_complex(x)
x = torch.tensor([[1.6116, -0.5772], [-1.4606, -0.912]])
result = torch.view_as_complex(input=x)
x = torch.tensor([0.4737 - 0.3839j, -0.2098 - 0.6699j, 0.347 - 0.9451j, -
    0.5174 - 1.3136j])
result = torch.view_as_real(x)
x = torch.tensor([0.4737 - 0.3839j, -0.2098 - 0.6699j, 0.347 - 0.9451j, -
    0.5174 - 1.3136j])
result = torch.view_as_real(input=x)
x = torch.tensor([[0.9383, -0.1983, 3.2, -1.2]])
y = 10.0
result = torch.where(x > 0, x, y)
x = torch.tensor([[0.9383, -0.1983, 3.2, -1.2]])
y = 10
result = torch.where(x > 0, x, y)
result = torch.zeros(3)
result = torch.zeros(3, 5)
input = torch.empty(2, 3)
result = torch.zeros_like(input)
result = torch.zeros_like(torch.empty(2, 3))
src = torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.uint8)
result = src.bfloat16().float()
src = torch.tensor([0.0 + 3.5j, -1 + 4.2j, 2.34 - 5.2j, -3.45 + 7.9j, -0.34 -
>>>>>>    8.2j, 0.23 + 9.2j, 1.0 + 1.0j, 2.0 + 0.5j, 3.0 - 1.0j], dtype=torch.
    complex64)
result = src.bfloat16().float()
src = torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.uint8)
result = src.bool()
src = torch.tensor([0.0 + 3.5j, -1 + 4.2j, 2.34 - 5.2j, -3.45 + 7.9j, -0.34 -
>>>>>>    8.2j, 0.23 + 9.2j, 1.0 + 1.0j, 2.0 + 0.5j, 3.0 - 1.0j], dtype=torch.
    complex64)
result = src.bool()
src = torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.uint8)
result = src.byte()
src = torch.tensor([0.0 + 3.5j, -1 + 4.2j, 2.34 - 5.2j, -3.45 + 7.9j, -0.34 -
>>>>>>    8.2j, 0.23 + 9.2j, 1.0 + 1.0j, 2.0 + 0.5j, 3.0 - 1.0j], dtype=torch.
    complex64)
result = src.byte()
src = torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.uint8)
result = src.cdouble()
src = torch.tensor([0.0 + 3.5j, -1 + 4.2j, 2.34 - 5.2j, -3.45 + 7.9j, -0.34 -
>>>>>>    8.2j, 0.23 + 9.2j, 1.0 + 1.0j, 2.0 + 0.5j, 3.0 - 1.0j], dtype=torch.
    complex64)
result = src.cdouble()
src = torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.uint8)
result = src.cfloat()
src = torch.tensor([0.0 + 3.5j, -1 + 4.2j, 2.34 - 5.2j, -3.45 + 7.9j, -0.34 -
>>>>>>    8.2j, 0.23 + 9.2j, 1.0 + 1.0j, 2.0 + 0.5j, 3.0 - 1.0j], dtype=torch.
    complex64)
result = src.cfloat()
src = torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.uint8)
result = src.char()
src = torch.tensor([0.0 + 3.5j, -1 + 4.2j, 2.34 - 5.2j, -3.45 + 7.9j, -0.34 -
>>>>>>    8.2j, 0.23 + 9.2j, 1.0 + 1.0j, 2.0 + 0.5j, 3.0 - 1.0j], dtype=torch.
    complex64)
result = src.char()
src = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
result = src.double()
src = torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.uint8)
result = src.double()
src = torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.uint8)
result = src.float()
src = torch.tensor([0.0 + 3.5j, -1 + 4.2j, 2.34 - 5.2j, -3.45 + 7.9j, -0.34 -
>>>>>>    8.2j, 0.23 + 9.2j, 1.0 + 1.0j, 2.0 + 0.5j, 3.0 - 1.0j], dtype=torch.
    complex64)
result = src.float()
src = torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.uint8)
result = src.half()
src = torch.tensor([0.0 + 3.5j, -1 + 4.2j, 2.34 - 5.2j, -3.45 + 7.9j, -0.34 -
>>>>>>    8.2j, 0.23 + 9.2j, 1.0 + 1.0j, 2.0 + 0.5j, 3.0 - 1.0j], dtype=torch.
    complex64)
result = src.half()
src = torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.uint8)
result = src.int()
src = torch.tensor([0.0 + 3.5j, -1 + 4.2j, 2.34 - 5.2j, -3.45 + 7.9j, -0.34 -
>>>>>>    8.2j, 0.23 + 9.2j, 1.0 + 1.0j, 2.0 + 0.5j, 3.0 - 1.0j], dtype=torch.
    complex64)
result = src.int()
src = torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.uint8)
result = src.long()
src = torch.tensor([0.0 + 3.5j, -1 + 4.2j, 2.34 - 5.2j, -3.45 + 7.9j, -0.34 -
>>>>>>    8.2j, 0.23 + 9.2j, 1.0 + 1.0j, 2.0 + 0.5j, 3.0 - 1.0j], dtype=torch.
    complex64)
result = src.long()
src = torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.uint8)
result = src.short()
src = torch.tensor([0.0 + 3.5j, -1 + 4.2j, 2.34 - 5.2j, -3.45 + 7.9j, -0.34 -
>>>>>>    8.2j, 0.23 + 9.2j, 1.0 + 1.0j, 2.0 + 0.5j, 3.0 - 1.0j], dtype=torch.
    complex64)
result = src.short()
result = torch.nn.init.calculate_gain('leaky_relu', 0.2)
result = torch.tensor(result)
result = torch.nn.init.calculate_gain(nonlinearity='relu', param=0.2)
result = torch.tensor(result)
conv = torch.nn.Conv2d(4, 6, (3, 3))
torch.nn.init.kaiming_normal_(conv.weight)
result = conv.weight
conv = torch.nn.Conv2d(4, 6, (3, 3))
torch.nn.init.kaiming_normal_(conv.weight)
result = conv.weight
conv = torch.nn.Conv2d(4, 6, (3, 3))
torch.nn.init.kaiming_uniform_(conv.weight)
result = conv.weight
conv = torch.nn.Conv2d(4, 6, (3, 3))
torch.nn.init.kaiming_uniform_(conv.weight)
result = conv.weight


class MyDataset(Dataset):

    def __init__(self, size=10):
        super(Dataset).__init__()
        self.data = list(range(size))

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


dataset = Subset(MyDataset(10), [1, 2, 3, 4, 5, 6])
result = []
for d in dataset:
    result.append(d)


class MyDataset(Dataset):

    def __init__(self, size=10):
        super(Dataset).__init__()
        self.data = list(range(size))

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


dataset = Subset(MyDataset(10), [9, 1])
result = []
for d in dataset:
    result.append(d)
a = torch.tensor([0.8, 0.1, 0.4])
result = a.bernoulli()
a = torch.ones(3, 3)
result = a.bernoulli()
a = torch.tensor([[0.0, 1.1, 1.2], [0.0, 0.0, 1.3], [0.0, 0.0, 0.0]])
result = a.count_nonzero()
a = torch.tensor([[0.0, 1.1, 1.2], [0.0, 0.0, 1.3], [0.0, 0.0, 0.0]])
result = a.count_nonzero(dim=1)
a = torch.tensor([4.0, 3.0])
b = torch.tensor([2.0, 2.0])
a.floor_divide_(b)
a = torch.tensor([4.0, 3.0])
b = torch.tensor([2.0, 2.0])
a.floor_divide_(other=b)
a = torch.tensor([1.0, 2, 3])
b = torch.tensor([4.0, 5, 6])
result = a.hypot(b)
a = torch.tensor([1.0])
b = torch.tensor([4.0, 5, 6])
result = a.hypot(other=b)
x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
result = x.kthvalue(4)
x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
result = x.kthvalue(2, 0, True)
x = torch.tensor([[0.56, 0.34, 0.78], [0.98, 0.21, 1.78]])
result = x.logcumsumexp(0)
x = torch.tensor([[0.56, 0.34, 0.78], [0.98, 0.21, 1.78]])
result = x.logcumsumexp(1)
input = torch.tensor([[[1, 2, 2], [2, 3, 3]], [[0, 5, 5], [9, 9, 0]]])
result, index = input.mode()
input = torch.tensor([[[1, 2, 2], [2, 3, 3]], [[0, 5, 5], [9, 9, 0]]])
result = input.mode()
result, index = result[0], result[1]
a = torch.tensor([0.4331, 1.2475, 0.6834, -0.2791])
a.pow_(2)
a = torch.tensor([0.4331, 1.2475, 0.6834, -0.2791])
a.pow_(exponent=3.0)
a = torch.tensor([-3.0, -2, -1, 1, 2, 3])
result = a.remainder_(torch.tensor(2.0))
result = torch.tensor([-3.0, -2, -1, 1, 2, 3]).remainder_(torch.tensor(2.0))
a = torch.arange(10)
split_sizes = [3, 2, 5]
result = a.split_with_sizes(split_sizes, dim=0)
a = torch.arange(6).reshape(2, 3)
split_sizes = [1, 2]
result = a.split_with_sizes(split_sizes, dim=1)
result = torch.zeros(2, 1, 2, 1, 2)
result.squeeze_()
result = torch.zeros(2, 1, 2, 1, 2).squeeze_()
result = torch.zeros(2, 2, 2)
result.unsqueeze_(0)
result = torch.zeros(2, 2, 1, 2).unsqueeze_(3)


class MyFunction(Function):

    @staticmethod
    def forward(ctx, x):
        ctx.x = x
        return x * 2

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.x
        grad_input = grad_output * 2
        return grad_input


data = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
output = MyFunction.apply(data)
output.backward(grad_tensor=torch.tensor([1.0, 1.0, 1.0]))
>>>>>>result = torch.utils.data.grad
result.requires_grad = False


class cus_func(Function):

    @staticmethod
    def forward(ctx, x):
        a = x + x
        b = x + x + x
        ctx.mark_non_differentiable(a)
        return a, b

    @staticmethod
    def backward(ctx, grad_a, grad_b):
        grad_x = 3 * grad_b
        return grad_x


data = torch.ones([2, 3], dtype=torch.float64, requires_grad=True)
a, b = cus_func.apply(data)
b.sum().backward()
>>>>>>result = torch.utils.data.grad


class cus_tanh(Function):

    @staticmethod
    def forward(ctx, x):
        y = torch.tanh(x)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, dy):
        grad = dy + 1
        return grad


data = torch.ones([2, 3], dtype=torch.float64, requires_grad=True)
z = cus_tanh.apply(data)
z.sum().backward()
>>>>>>result = torch.utils.data.grad


class cus_tanh(Function):

    @staticmethod
    def forward(ctx, x):
        ctx.set_materialize_grads(False)
        return x + x + x, x + x

    @staticmethod
    def backward(ctx, grad, grad2):
        assert grad2 == None
        return grad


x = torch.ones([1], dtype=torch.float64)
x.requires_grad = True
cus_tanh.apply(x)[0].backward()
result = x.grad
result.requires_grad = False


class cus_tanh(Function):

    @staticmethod
    def forward(ctx, x):
        ctx.set_materialize_grads(value=False)
        return x + x + x, x + x

    @staticmethod
    def backward(ctx, grad, grad2):
        assert grad2 == None
        return grad


x = torch.ones([1], dtype=torch.float64)
x.requires_grad = True
cus_tanh.apply(x)[0].backward()
result = x.grad
result.requires_grad = False
a = torch.tensor([0.8, 0.1, 0.4])
result = torch.bernoulli(a)
a = torch.ones(3, 3)
result = torch.bernoulli(a)
x = torch.tensor([1, 2, 3], dtype=torch.int32)
result = torch.combinations(input=x)
x = torch.tensor([1, 2, 3], dtype=torch.int32)
result = torch.combinations(x)
a = torch.tensor([[0.0, 1.1, 1.2], [0.0, 0.0, 1.3], [0.0, 0.0, 0.0]])
result = torch.count_nonzero(a)
a = torch.tensor([[0.0, 1.1, 1.2], [0.0, 0.0, 1.3], [0.0, 0.0, 0.0]])
result = torch.count_nonzero(input=a, dim=1)
result = torch.cumulative_trapezoid(torch.tensor([1.0, 1, 1, 0, 1]))
y = torch.tensor([1, 1, 1, 0, 1]).astype(torch.float32)
x = torch.tensor([1, 2, 3, 0, 1]).astype(torch.float32)
result = torch.cumulative_trapezoid(y, x)
x = torch.tensor([10.0, -2.5, 0.0, 3.14])
result, exponent = torch.frexp(x)
x = torch.tensor([[128.0, 64.0], [-32.0, 16.0]])
result, ex = torch.frexp(x)
a = torch.tensor([1.0, 2, 3])
b = torch.tensor([4.0, 5, 6])
result = torch.hypot(a, b)
a = torch.tensor([1.0])
b = torch.tensor([4.0, 5, 6])
result = torch.hypot(input=a, other=b)
x = torch.tensor([-float('inf'), float('inf'), 1.2])
result = torch.isneginf(x)
out = torch.tensor([False, False, False])
x = torch.tensor([-float('inf'), float('inf'), 1.2])
result = torch.isneginf(input=x, out=out)
x = torch.tensor([-float('inf'), float('inf'), 1.2])
result = torch.isposinf(x)
out = torch.tensor([False, False, False])
x = torch.tensor([-float('inf'), float('inf'), 1.2])
result = torch.isposinf(input=x, out=out)
result = torch.isreal(torch.tensor([1, 1 + 1.0j, 2 + 0.0j]))
result = torch.isreal(input=torch.tensor([-0.0, -2.1, 2.5]))
mat1 = torch.eye(2)
mat2 = torch.ones(2, 2)
result = torch.kron(mat1, mat2)
mat1 = torch.eye(2)
mat2 = torch.ones(2, 2)
result = torch.kron(input=mat1, other=mat2)
x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
result = torch.kthvalue(x, 4)
x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
result = torch.kthvalue(x, 2, 0, True)
x = torch.tensor([[0.56, 0.34, 0.78], [0.98, 0.21, 1.78]])
result = torch.logcumsumexp(x, 0)
x = torch.tensor([[0.56, 0.34, 0.78], [0.98, 0.21, 1.78]])
result = torch.logcumsumexp(x, 1)
input = torch.tensor([[[1, 2, 2], [2, 3, 3]], [[0, 5, 5], [9, 9, 0]]])
result, index = torch.mode(input)
input = torch.tensor([[[1, 2, 2], [2, 3, 3]], [[0, 5, 5], [9, 9, 0]]])
result = torch.mode(input)
result, index = result[0], result[1]
a = torch.tensor([[1.0, 2.0], [4.0, 5.0]])
b = torch.tensor([1.0, 3.0])
result = torch.mv(a, b)
a = torch.tensor([[1.0, 2.0], [4.0, 5.0]])
b = torch.tensor([1.0, 3.0])
result = torch.mv(input=a, vec=b)


class SubModel(nn.Module):

    def __init__(self):
        super(SubModel, self).__init__()
        self.register_buffer('buf1', torch.tensor([1.0, 2.0, 4.0, 5.0]))
        self.register_buffer('buf4', torch.tensor([1.0, 2.0, 4.0, 5.0]))
        self.register_buffer('buf5', torch.tensor([1.0, 2.0, 4.0, 5.0]))

    def forward(self, x):
        return x


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.sub = SubModel()
        self.register_buffer('buf1', torch.tensor([1.0, 2.0, 4.0, 5.0]))
        self.register_buffer('buf2', torch.tensor([1.0, 2.0, 4.0, 5.0]))
        self.register_buffer('buf3', torch.tensor([1.0, 2.0, 4.0, 5.0]))

    def forward(self, x):
        return x


model = Model()
result = []
for name, buf in model.named_buffers():
    result.append((name, buf))


class SubModel(nn.Module):

    def __init__(self):
        super(SubModel, self).__init__()
        self.register_buffer('buf1', torch.tensor([1.0, 2.0, 4.0, 5.0]))
        self.register_buffer('buf4', torch.tensor([1.0, 2.0, 4.0, 5.0]))
        self.register_buffer('buf5', torch.tensor([1.0, 2.0, 4.0, 5.0]))

    def forward(self, x):
        return x


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.sub = SubModel()
        self.register_buffer('buf1', torch.tensor([1.0, 2.0, 4.0, 5.0]))
        self.register_buffer('buf2', torch.tensor([1.0, 2.0, 4.0, 5.0]))
        self.register_buffer('buf3', torch.tensor([1.0, 2.0, 4.0, 5.0]))

    def forward(self, x):
        return x


model = Model()
result = []
for name, buf in model.named_buffers(prefix='wfs'):
    result.append((name, buf))
choices = nn.ParameterDict({f'param_{i}': nn.Parameter(torch.ones(i + 1, i +
    1)) for i in range(10)})
result = list(choices)
choices = nn.ParameterDict()
result = list(choices)
modified_backend_state = [torch.nn.attention.SDPBackend.MATH]
np.random.seed(100)
x_data = np.random.randn(2, 2, 2, 2)
x = torch.tensor(x_data, dtype=torch.float32)
with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
    result = paddle.compat.nn.functional.scaled_dot_product_attention(x, x, x
        ).to(torch.float32)
    current_backends = torch.nn.attention._cur_sdpa_kernel_backends()
    assert current_backends == modified_backend_state
original_backend_state = set(torch.nn.attention._cur_sdpa_kernel_backends())
modified_backend_state = [torch.nn.attention.SDPBackend.MATH]
np.random.seed(100)
x_data = np.random.randn(2, 2, 2, 2)
x = torch.tensor(x_data, dtype=torch.float32)
current_backends = set(torch.nn.attention._cur_sdpa_kernel_backends())
assert current_backends == original_backend_state, f'Expected {original_backend_state}, got {current_backends}'
with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
    output1 = paddle.compat.nn.functional.scaled_dot_product_attention(x, x, x
        ).to(torch.float32)
    current_backends = torch.nn.attention._cur_sdpa_kernel_backends()
    assert current_backends == modified_backend_state, f'Expected {modified_backend_state}, got {current_backends}'
    output2 = paddle.compat.nn.functional.scaled_dot_product_attention(x, x, x
        ).to(torch.float32)
    current_backends = torch.nn.attention._cur_sdpa_kernel_backends()
    assert current_backends == modified_backend_state, f'Expected {modified_backend_state}, got {current_backends}'
    output3 = paddle.compat.nn.functional.scaled_dot_product_attention(x, x, x
        ).to(torch.float32)
    current_backends = torch.nn.attention._cur_sdpa_kernel_backends()
    assert current_backends == modified_backend_state, f'Expected {modified_backend_state}, got {current_backends}'
current_backends = set(torch.nn.attention._cur_sdpa_kernel_backends())
assert current_backends == original_backend_state, f'Expected {original_backend_state}, got {current_backends}'
result = output1 + output2 + output3
modified_backend_state = {torch.nn.attention.SDPBackend.MATH, torch.nn.
    attention.SDPBackend.FLASH_ATTENTION}
np.random.seed(100)
x_data = np.random.randn(2, 2)
x = torch.tensor(x_data, dtype=torch.float32)
with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH,
    torch.nn.attention.SDPBackend.FLASH_ATTENTION]):
    x = x + 1
    current_backends = set(torch.nn.attention._cur_sdpa_kernel_backends())
    assert current_backends == modified_backend_state, f'Expected {modified_backend_state}, got {current_backends}'
    x = x + 1
result = x
modified_backend_state = [torch.nn.attention.SDPBackend.MATH]
np.random.seed(100)
x_data = np.random.randn(2, 2, 2, 2)
x = torch.tensor(x_data, dtype=torch.float32)
with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
    result = paddle.compat.nn.functional.scaled_dot_product_attention(x, x, x
        ).to(torch.float32)
    current_backends = torch.nn.attention._cur_sdpa_kernel_backends()
    assert current_backends == modified_backend_state
original_backend_state = set(torch.nn.attention._cur_sdpa_kernel_backends())
modified_backend_state = [torch.nn.attention.SDPBackend.MATH]
np.random.seed(100)
x_data = np.random.randn(2, 2, 2, 2)
x = torch.tensor(x_data, dtype=torch.float32)
current_backends = set(torch.nn.attention._cur_sdpa_kernel_backends())
assert current_backends == original_backend_state, f'Expected {original_backend_state}, got {current_backends}'
with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
    output1 = paddle.compat.nn.functional.scaled_dot_product_attention(x, x, x
        ).to(torch.float32)
    current_backends = torch.nn.attention._cur_sdpa_kernel_backends()
    assert current_backends == modified_backend_state, f'Expected {modified_backend_state}, got {current_backends}'
    output2 = paddle.compat.nn.functional.scaled_dot_product_attention(x, x, x
        ).to(torch.float32)
    current_backends = torch.nn.attention._cur_sdpa_kernel_backends()
    assert current_backends == modified_backend_state, f'Expected {modified_backend_state}, got {current_backends}'
    output3 = paddle.compat.nn.functional.scaled_dot_product_attention(x, x, x
        ).to(torch.float32)
    current_backends = torch.nn.attention._cur_sdpa_kernel_backends()
    assert current_backends == modified_backend_state, f'Expected {modified_backend_state}, got {current_backends}'
current_backends = set(torch.nn.attention._cur_sdpa_kernel_backends())
assert current_backends == original_backend_state, f'Expected {original_backend_state}, got {current_backends}'
result = output1 + output2 + output3
modified_backend_state = [torch.nn.attention.SDPBackend.MATH]
np.random.seed(100)
x_data = np.random.randn(2, 2, 2, 2)
x = torch.tensor(x_data, dtype=torch.float32)
with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
    result = paddle.compat.nn.functional.scaled_dot_product_attention(x, x, x
        ).to(torch.float32)
    current_backends = torch.nn.attention._cur_sdpa_kernel_backends()
    assert current_backends == modified_backend_state
original_backend_state = set(torch.nn.attention._cur_sdpa_kernel_backends())
modified_backend_state = [torch.nn.attention.SDPBackend.MATH]
np.random.seed(100)
x_data = np.random.randn(2, 2, 2, 2)
x = torch.tensor(x_data, dtype=torch.float32)
current_backends = set(torch.nn.attention._cur_sdpa_kernel_backends())
assert current_backends == original_backend_state, f'Expected {original_backend_state}, got {current_backends}'
with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
    output1 = paddle.compat.nn.functional.scaled_dot_product_attention(x, x, x
        ).to(torch.float32)
    current_backends = torch.nn.attention._cur_sdpa_kernel_backends()
    assert current_backends == modified_backend_state, f'Expected {modified_backend_state}, got {current_backends}'
    output2 = paddle.compat.nn.functional.scaled_dot_product_attention(x, x, x
        ).to(torch.float32)
    current_backends = torch.nn.attention._cur_sdpa_kernel_backends()
    assert current_backends == modified_backend_state, f'Expected {modified_backend_state}, got {current_backends}'
    output3 = paddle.compat.nn.functional.scaled_dot_product_attention(x, x, x
        ).to(torch.float32)
    current_backends = torch.nn.attention._cur_sdpa_kernel_backends()
    assert current_backends == modified_backend_state, f'Expected {modified_backend_state}, got {current_backends}'
current_backends = set(torch.nn.attention._cur_sdpa_kernel_backends())
assert current_backends == original_backend_state, f'Expected {original_backend_state}, got {current_backends}'
result = output1 + output2 + output3
input = torch.randn(1, 9, 4, 4)
result = F.pixel_shuffle(input, 3)
input = torch.randn(1, 9, 4, 4)
result = F.pixel_shuffle(input, upscale_factor=3)
input = torch.randn(1, 1, 12, 12)
result = F.pixel_unshuffle(input, 3)
input = torch.randn(1, 1, 12, 12)
result = F.pixel_unshuffle(input, downscale_factor=3)
rates = torch.rand(4, 4) * 5
result = torch.poisson(rates)
rates = torch.tensor([[1.0, 3.0, 4.0], [2.0, 3.0, 6.0]])
result = torch.poisson(rates)
y = torch.tensor([1.0, 1, 1, 0, 1])
result = torch.trapezoid(y)
y = torch.tensor([1, 1, 1, 0, 1]).astype(torch.float32)
x = torch.tensor([1, 2, 3, 0, 1]).astype(torch.float32)
result = torch.trapezoid(y=y, x=x)


class MyIterableDataset(torch.utils.data.IterableDataset):

    def __init__(self, start, end):
        super(MyIterableDataset).__init__()
        assert end > start, 'this example code only works with end >= start'
        self.start = start
        self.end = end

    def __iter__(self):
        iter_start = self.start
        iter_end = self.end
        return iter(range(iter_start, iter_end))


dataset = torch.utils.data.ChainDataset([MyIterableDataset(start=3, end=7),
    MyIterableDataset(start=3, end=7)])
result = []
for d in dataset:
    result.append(d)


class MyIterableDataset(data.IterableDataset):

    def __init__(self, start, end):
        super(MyIterableDataset).__init__()
        assert end > start, 'this example code only works with end >= start'
        self.start = start
        self.end = end

    def __iter__(self):
        iter_start = self.start
        iter_end = self.end
        return iter(range(iter_start, iter_end))


dataset = data.ChainDataset([MyIterableDataset(start=1, end=10),
    MyIterableDataset(start=1, end=3)])
result = []
for d in dataset:
    result.append(d)


class MyIterableDataset(IterableDataset):

    def __init__(self, start, end):
        super(MyIterableDataset).__init__()
        assert end > start, 'this example code only works with end >= start'
        self.start = start
        self.end = end

    def __iter__(self):
        return iter(range(self.start, self.end))


ds = MyIterableDataset(start=3, end=7)
result = []
for i in ds:
    result.append(i)


class MyIterableDataset(data.IterableDataset):

    def __init__(self, start, end):
        super(MyIterableDataset).__init__()
        assert end > start, 'this example code only works with end >= start'
        self.start = start
        self.end = end

    def __iter__(self):
        return iter(range(self.start, self.end))


ds = MyIterableDataset(start=3, end=7)
result = next(ds.__iter__())
result = torch.tensor(default_collate([0, 1, 2, 3]))
result = default_collate(['a', 'b', 'c'])
x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
result = x.kron(y)
x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
result = torch.Tensor.kron(x, y)
x = torch.empty(2, 3)
result = x.log_normal_()
x = torch.empty(2, 3)
result = x.log_normal_(1.0, 0.5)
op_type = torch.distributed.ReduceOp
op_type = ReduceOp
ops = []
result = torch.distributed.batch_isend_irecv(ops)
ops = []
result = torch.distributed.batch_isend_irecv(ops, op=torch.distributed.
    ReduceOp.MIN)
result = torch.distributed.get_backend()
result = torch.distributed.get_backend(group=None)
result = torch.distributed.get_rank()
result = torch.distributed.get_rank(group=None)
result = torch.distributed.get_world_size()
result = torch.distributed.get_world_size(group=None)
mode = torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION
mode = SDPBackend.EFFICIENT_ATTENTION
mode = torch.nn.attention.SDPBackend.ERROR
mode = SDPBackend.ERROR
cls = transformers.PreTrainedTokenizer
tokenizer = transformers.PreTrainedTokenizer()
value = torch.distributed.ReduceOp.MAX
value = ReduceOp.MAX
value = torch.distributed.ReduceOp.MIN
value = ReduceOp.MIN
value = torch.distributed.ReduceOp.SUM
value = ReduceOp.SUM
module = torch.nn
module = nn
