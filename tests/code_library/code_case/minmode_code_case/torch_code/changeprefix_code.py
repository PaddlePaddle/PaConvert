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
#

import torch
import copy
import pickle
import numpy
import torch.nn as nn
from torch.autograd import Function
from collections import OrderedDict
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
from torch.utils.cpp_extension import BuildExtension, CppExtension
from torch.utils.cpp_extension import CUDA_HOME
import torch.utils.data as torch_data
from torch.utils.data import Dataset, ConcatDataset
from torch.utils.data import Sampler
from torch.utils.data import SequentialSampler
from torch.utils.data import IterableDataset, ChainDataset, get_worker_info
from torch.utils.data import Subset
from torch.utils.data import default_collate
from torch.distributed import ReduceOp
from torch.nn.attention import SDPBackend
import transformers

# torch.BFloat16Tensor
result = torch.BFloat16Tensor([1.5, 2, 3])

result = torch.BFloat16Tensor()

# torch.BoolTensor
result = torch.BoolTensor(2, 3)

shape = [2, 3]
result = torch.BoolTensor(*shape)

# torch.ByteTensor
result = torch.ByteTensor(2, 3)

shape = [2, 3]
result = torch.ByteTensor(*shape)

# torch.CharTensor
result = torch.CharTensor(2, 3)

shape = [2, 3]
result = torch.CharTensor(*shape)

# torch.DoubleTensor
result = torch.DoubleTensor(2, 3)

shape = [2, 3]
result = torch.DoubleTensor(*shape)

# torch.FloatTensor
result = torch.FloatTensor(2, 3)

shape = [2, 3]
result = torch.FloatTensor(*shape)

# torch.Generator
result = torch.Generator(device='cpu')

result = torch.Generator()

# torch.HalfTensor
result = torch.HalfTensor(2, 3)

shape = [2, 3]
result = torch.HalfTensor(*shape)

# torch.IntTensor
result = torch.IntTensor(2, 3)

shape = [2, 3]
result = torch.IntTensor(*shape)

# torch.LongTensor
result = torch.LongTensor(2, 3)

shape = [2, 3]
result = torch.LongTensor(*shape)

# torch.ShortTensor
result = torch.ShortTensor(2, 3)

shape = [2, 3]
result = torch.ShortTensor(*shape)

# torch.Size
result = list(torch.Size([2, 8, 64, 64]))

result = torch.randn(6, 5, 7).size() == torch.Size([6, 5, 7])

# torch.Tensor
result = torch.Tensor(2, 3)

shape = [2, 3]
result = torch.Tensor(*shape)

# torch.Tensor.T
x = torch.arange(16).reshape(4, 4)
result = x.T

result = torch.arange(16).reshape(4, 4).T

# torch.Tensor.__add__
x = torch.tensor([1, 2, 3], dtype=torch.int64)
y = torch.tensor([3, 2, 1], dtype=torch.int64)
result = x + y

x = torch.tensor([1., 2., 3.], dtype=torch.float32)
y = torch.tensor([3., 2., 1.], dtype=torch.float32)
result = x + y

# torch.Tensor.__and__
x = torch.tensor([True, False, True])
y = torch.tensor([True, True, False])
result = x & y

x = torch.tensor([1, 2, 3], dtype=torch.int32)
y = torch.tensor([3, 2, 1], dtype=torch.int32)
result = x & y

# torch.Tensor.__array__
x = torch.tensor([1.0, 2.0, 3.0])
result = x.__array__()

x = torch.tensor([[1, 2], [3, 4]])
result = x.__array__()

# torch.Tensor.__bool__
x = torch.tensor([True])
result = bool(x)

x = torch.tensor([0.])
result = bool(x)

# torch.Tensor.__deepcopy__
x = torch.tensor([True])
result = copy.deepcopy(x)

x = torch.tensor([0.])
result = copy.deepcopy(x)

# torch.Tensor.__eq__
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([1.0, 2.0, 3.0])
result = x == y

x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([1.0, 5.0, 3.0])
result = x == y

# torch.Tensor.__floordiv__
x = torch.tensor([2.0, 4.0])
result = 8 // x

x = torch.tensor([1, 2])
result = 10 // x

# torch.Tensor.__format__
x = torch.tensor(3.14159)
result = format(x, '.2f')

x = torch.tensor(123)
result = format(x, '05d')

# torch.Tensor.__ge__
x = torch.tensor([3.0, 2.0, 1.0])
y = torch.tensor([1.0, 2.0, 3.0])
result = x >= y

x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([1.0, 5.0, -1.0])
result = x >= y

# torch.Tensor.__getitem__
x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
result = x[1, 2]

x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
result = x[:, 1:3]

# torch.Tensor.__gt__
x = torch.tensor([3.0, 2.0, 1.0])
y = torch.tensor([1.0, 2.0, 3.0])
result = x > y

x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([1.0, 5.0, -1.0])
result = x > y

# torch.Tensor.__index__
x = torch.tensor(5)
result = x.__index__()

x = torch.tensor(3, dtype=torch.int64)
result = x.__index__()

# torch.Tensor.__int__
x = torch.tensor(2.0)
result = int(x)

x = torch.tensor([2.0])
result = int(x)

# torch.Tensor.__invert__
x = torch.tensor([True, False])
result = ~x

x = torch.tensor([1, 2], dtype=torch.int32)
result = ~x

# torch.Tensor.__ior__
x = torch.tensor([True, False, True])
y = torch.tensor([True, True, False])
x |= y

x = torch.tensor([1, 2, 3], dtype=torch.int32)
y = torch.tensor([3, 2, 1], dtype=torch.int32)
x |= y

# torch.Tensor.__le__
x = torch.tensor([3.0, 2.0, 1.0])
y = torch.tensor([1.0, 2.0, 3.0])
result = x <= y

x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([1.0, 5.0, -1.0])
result = x <= y

# torch.Tensor.__len__
x = torch.tensor([1.0, 2.0, 3.0, 4.0])
result = len(x)

x = torch.tensor([[1, 2], [3, 4], [5, 6]])
result = len(x)

# torch.Tensor.__lt__
x = torch.tensor([3.0, 2.0, 1.0])
y = torch.tensor([1.0, 2.0, 3.0])
result = x < y

x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([1.0, 5.0, -1.0])
result = x < y

# torch.Tensor.__mul__
x = torch.tensor([1, 2, 3], dtype=torch.int64)
y = torch.tensor([3, 2, 1], dtype=torch.int64)
result = x * y

x = torch.tensor([1., 2., 3.], dtype=torch.float32)
y = torch.tensor([3., 2., 1.], dtype=torch.float32)
result = x * y

# torch.Tensor.__ne__
x = torch.tensor([3.0, 2.0, 1.0])
y = torch.tensor([1.0, 2.0, 3.0])
result = x != y

x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([1.0, 5.0, -1.0])
result = x != y

# torch.Tensor.__neg__
x = torch.tensor([1., 2.], dtype=torch.float32)
result = -x

x = torch.tensor([3., -1.], dtype=torch.float64)
result = -x

# torch.Tensor.__not__
x = torch.tensor([1.], dtype=torch.float32)
result = not x

x = torch.tensor([3.], dtype=torch.float64)
result = not x

# torch.Tensor.__or__
x = torch.tensor([True, False, True])
y = torch.tensor([True, True, False])
result = x | y

x = torch.tensor([1, 2, 3], dtype=torch.int32)
y = torch.tensor([3, 2, 1], dtype=torch.int32)
result = x | y

# torch.Tensor.__pow__
x = torch.tensor([2.0, 3.0])
result = x ** 2

x = torch.tensor([1.0, 2.0])
result = x ** 3.0

# torch.Tensor.__radd__
x = torch.tensor([1, 2, 3], dtype=torch.int64)
result = 5 + x

x = torch.tensor([1., 2., 3.], dtype=torch.float32)
result = 5. + x

# torch.Tensor.__reduce_ex__
x = torch.tensor([1, 2, 3], device=torch.device('cpu'), dtype=torch.int64, requires_grad=False)
data = pickle.dumps(x)
result = pickle.loads(data)

x = torch.tensor([1, 2, 3], device=torch.device('cuda'), dtype=torch.int64, pin_memory=False, requires_grad=False)
data = pickle.dumps(x)
result = pickle.loads(data)

# torch.Tensor.__rmul__
x = torch.tensor([1, 2, 3], dtype=torch.int64)
result = 5 * x

x = torch.tensor([1., 2., 3.], dtype=torch.float32)
result = 5. * x

# torch.Tensor.__rpow__
x = torch.tensor([2.0, 3.0])
result = 2 ** x

x = torch.tensor([1.0, 2.0])
result = 3.0 ** x

# torch.Tensor.__rsub__
x = torch.tensor([2.0, 3.0])
result = 5 - x

x = torch.tensor([1, 2])
result = 10 - x

# torch.Tensor.__rtruediv__
x = torch.tensor([2.0, 4.0])
result = 8 / x

x = torch.tensor([1, 2])
result = 10 / x

# torch.Tensor.__setitem__
x = torch.tensor([1.0, 2.0, 3.0])
x[1] = 5.0
result = x

x = torch.tensor([[1, 2], [3, 4]])
x[0, :] = torch.tensor([5, 6])
result = x

# torch.Tensor.__sub__
x = torch.tensor([1, 2, 3], dtype=torch.int64)
y = torch.tensor([3, 2, 1], dtype=torch.int64)
result = x - y

x = torch.tensor([1., 2., 3.], dtype=torch.float32)
y = torch.tensor([3., 2., 1.], dtype=torch.float32)
result = x - y

# torch.Tensor.__xor__
x = torch.tensor([True, False, True])
y = torch.tensor([True, True, False])
result = x ^ y

x = torch.tensor([1, 2, 3], dtype=torch.int32)
y = torch.tensor([3, 2, 1], dtype=torch.int32)
result = x ^ y

# torch.Tensor.abs
a = torch.tensor([[-4, 9], [-23, 2]])
result = a.abs()

result = torch.tensor([[-4, 9], [-23, 2]]).abs()

# torch.Tensor.abs_
a = torch.tensor([-1])
a.abs_()

a = torch.tensor([-1, -2, 3])
a.abs_()

# torch.Tensor.acos
a = torch.tensor([[ 0.3348, -0.5889,  0.2005, -0.1584], [ 0.3348, -0.5889,  0.2005, -0.1584]])
result = a.acos()

result = torch.tensor([[ 0.3348, -0.5889,  0.2005, -0.1584]]).acos()

# torch.Tensor.acos_
a = torch.tensor([0.34, -0.56, 0.73])
a.acos_()

a = torch.tensor([1., -1., 0.])
a.acos_()

# torch.Tensor.acosh
result = torch.tensor([1.3192, 1.9915, 1.9674, 1.7151]).acosh()

a = torch.tensor([1.3192, 1.9915, 1.9674, 1.7151])
result = a.acosh()

# torch.Tensor.acosh_
result = torch.tensor([1.3192, 1.9915, 1.9674, 1.7151]).acosh_()

a = torch.tensor([1.3192, 1.9915, 1.9674, 1.7151])
result = a.acosh_()

# torch.Tensor.add
x = torch.tensor([1, 2, 3])
result = x.add(torch.tensor([1, 4, 6]))

x = torch.tensor([1, 2, 3])
result = x.add(20)

# torch.Tensor.add_
x = torch.tensor([1, 2, 3])
x.add_(torch.tensor([1, 4, 6]))

x = torch.tensor([1, 2, 3])
x.add_(20)

# torch.Tensor.addmm
x = torch.tensor([[1, 2], [4, 5]])
mat1 = torch.tensor([[1, 2], [4, 5]])
mat2 = torch.tensor([[1, 2], [4, 5]])
result = x.addmm(mat1, mat2)

x = torch.tensor([[1., 2], [4, 5]])
mat1 = torch.tensor([[1., 2], [4, 5]])
mat2 = torch.tensor([[1., 2], [4, 5]])
result = x.addmm(mat1, mat2, beta=0.6, alpha=0.7)

# torch.Tensor.addmm_
x = torch.tensor([[1, 2], [4, 5]])
mat1 = torch.tensor([[1, 2], [4, 5]])
mat2 = torch.tensor([[1, 2], [4, 5]])
x.addmm_(mat1, mat2)

x = torch.tensor([[1., 2], [4, 5]])
mat1 = torch.tensor([[1., 2], [4, 5]])
mat2 = torch.tensor([[1., 2], [4, 5]])
x.addmm_(mat1, mat2, beta=0.6, alpha=0.7)

# torch.Tensor.all
a = torch.rand(1, 2).bool()
result = a.all()

a = torch.rand(3, 4)
result = a.all()

# torch.Tensor.amax
x = torch.tensor([[1, 2, 3], [3, 4, 6]])
result = x.amax()

x = torch.tensor([[1, 2, 3], [3, 4, 6]])
result = x.amax(dim=1)

# torch.Tensor.amin
x = torch.tensor([[1, 2, 3], [3, 4, 6]])
result = x.amin()

x = torch.tensor([[1, 2, 3], [3, 4, 6]])
result = x.amin(dim=1)

# torch.Tensor.angle
result = torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j]).angle()

x = torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j])
result = x.angle() * 180 / 3.14159

# torch.Tensor.any
a = torch.rand(1, 2).bool()
result = a.any()

a = torch.rand(3, 4)
result = a.any()

# torch.Tensor.apply_
x = torch.tensor([1.3192, 1.9915, 1.9674, 1.7151])
result = x.apply_(lambda x: x*2)

# torch.Tensor.argmax
x = torch.tensor([[1, 2, 3], [3, 4, 6]])
result = x.argmax()

x = torch.tensor([[1, 2, 3], [3, 4, 6]])
result = x.argmax(dim=1)

# torch.Tensor.argmin
x = torch.tensor([[1, 2, 3], [3, 4, 6]])
result = x.argmin()

x = torch.tensor([[1, 2, 3], [3, 4, 6]])
result = x.argmin(dim=1)

# torch.Tensor.argsort
input = torch.tensor([[ 1.3398,  0.2663, -0.2686,  0.2450],
                    [-0.7401, -0.8805, -0.3402, -1.1936],
                    [ 0.4907, -1.3948, -1.0691, -0.3132],
                    [-1.6092,  0.5419, -0.2993,  0.3195]])
result = input.argsort()

input = torch.tensor([[ 1.3398,  0.2663, -0.2686,  0.2450],
                    [-0.7401, -0.8805, -0.3402, -1.1936],
                    [ 0.4907, -1.3948, -1.0691, -0.3132],
                    [-1.6092,  0.5419, -0.2993,  0.3195]])
result = input.argsort(dim = 1)

# torch.Tensor.as_strided
x = torch.tensor([[ 0.0335,  0.1830, -0.1269],
[ 0.1897, -0.1422, -0.4940],
[-0.7674, -0.0134, -0.3733]])
results = x.as_strided((2, 2), (1, 2))

x = torch.tensor([[ 0.0335,  0.1830, -0.1269],
[ 0.1897, -0.1422, -0.4940],
[-0.7674, -0.0134, -0.3733]])
results = x.as_strided((2, 2), (1, 2), 0)

# torch.Tensor.asin
result = torch.tensor([0.34, -0.56, 0.73]).asin()

a = torch.tensor([0.34, -0.56, 0.73])
result = a.asin()

# torch.Tensor.asin_
result = torch.tensor([0.34, -0.56, 0.73]).asin_()

a = torch.tensor([0.34, -0.56, 0.73])
result = a.asin_()

# torch.Tensor.asinh
a = torch.Tensor([[1.,2.], [3.,4.]])
result = a.asinh()

# torch.Tensor.asinh_
result = torch.tensor([0.34, -0.56, 0.73]).asinh_()

a = torch.tensor([0.34, -0.56, 0.73])
result = a.asinh_()

# torch.Tensor.atan
result = torch.tensor([0.34, -0.56, 0.73]).atan()

a = torch.tensor([0.34, -0.56, 0.73])
result = a.atan()

# torch.Tensor.atan2
input = torch.tensor([ 0.9041,  0.0196, -0.3108, -2.4423])
other = torch.tensor([ 0.2341,  0.2539, -0.6256, -0.6448])
result = input.atan2(other)

result = torch.tensor([ 0.9041,  0.0196, -0.3108, -2.4423]).atan2(torch.tensor([ 0.2341,  0.2539, -0.6256, -0.6448]))

# torch.Tensor.atan_
result = torch.tensor([0.34, -0.56, 0.73]).atan_()

a = torch.tensor([0.34, -0.56, 0.73])
result = a.atan_()

# torch.Tensor.atanh
a = torch.Tensor([[1.,2.], [3.,4.]])
result = a.atanh()

# torch.Tensor.atanh_
result = torch.tensor([0.34, -0.56, 0.73]).atanh_()

a = torch.tensor([0.34, -0.56, 0.73])
result = a.atanh_()

# torch.Tensor.baddbmm
a = torch.tensor([[[4., 5., 6.], [1., 2., 3.]]])
b = torch.tensor([[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]])
input = torch.tensor([[[1., 2., 3.], [4., 5., 6.]]])
result = input.baddbmm(a, b)

a = torch.tensor([[[4., 5., 6.], [1., 2., 3.]]])
b = torch.tensor([[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]])
input = torch.tensor([[[1., 2., 3.], [4., 5., 6.]]])
result = input.baddbmm(a, b, beta=3)

# torch.Tensor.baddbmm_
a = torch.tensor([[[4., 5., 6.], [1., 2., 3.]]])
b = torch.tensor([[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]])
input = torch.tensor([[[1., 2., 3.], [4., 5., 6.]]])
result = input.baddbmm_(a, b)

a = torch.tensor([[[4., 5., 6.], [1., 2., 3.]]])
b = torch.tensor([[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]])
input = torch.tensor([[[1., 2., 3.], [4., 5., 6.]]])
result = input.baddbmm_(a, b, beta=3)

# torch.Tensor.bincount
input = torch.tensor([4, 3, 6, 3, 4])
weights = torch.tensor([ 0.0000,  0.2500,  0.5000,  0.7500,  1.0000])
result = input.bincount()

input = torch.tensor([4, 3, 6, 3, 4])
weights = torch.tensor([ 0.0000,  0.2500,  0.5000,  0.7500,  1.0000])
result = input.bincount(weights)

# torch.Tensor.bitwise_and
x = torch.tensor([1, 2, 3])
y = torch.tensor([-1, 9, 10])
result = x.bitwise_and(y)

x = torch.tensor([True, False, True])
y = torch.tensor([False, False, True])
result = x.bitwise_and(y)

# torch.Tensor.bitwise_and_
x = torch.tensor([1, 2, 3])
y = torch.tensor([-1, 9, 10])
x.bitwise_and_(y)

x = torch.tensor([True, False, True])
y = torch.tensor([False, False, True])
x.bitwise_and_(y)

# torch.Tensor.bitwise_left_shift
input = torch.tensor([-2, -7, 31], dtype=torch.int8)
other = torch.tensor([1, 0, 3], dtype=torch.int8)
result = input.bitwise_left_shift(other)

input = torch.tensor([-2, -7, 31], dtype=torch.int8)
other = torch.tensor([1, 0, 3], dtype=torch.int8)
result = input.bitwise_left_shift(other=other)

# torch.Tensor.bitwise_left_shift_
input = torch.tensor([-2, -7, 31], dtype=torch.int8)
other = torch.tensor([1, 0, 3], dtype=torch.int8)
result = input.bitwise_left_shift_(other)

input = torch.tensor([-2, -7, 31], dtype=torch.int8)
other = torch.tensor([1, 0, 3], dtype=torch.int8)
result = input.bitwise_left_shift_(other=other)

# torch.Tensor.bitwise_not
x = torch.tensor([1, 2, 3])
result = x.bitwise_not()

x = torch.tensor([True, False, True])
result = x.bitwise_not()

# torch.Tensor.bitwise_not_
x = torch.tensor([1, 2, 3])
x.bitwise_not_()

x = torch.tensor([True, False, True])
x.bitwise_not_()

# torch.Tensor.bitwise_or
x = torch.tensor([1, 2, 3])
y = torch.tensor([-1, 9, 10])
result = x.bitwise_or(y)

x = torch.tensor([True, False, True])
y = torch.tensor([False, False, True])
result = x.bitwise_or(y)

# torch.Tensor.bitwise_or_
x = torch.tensor([1, 2, 3])
y = torch.tensor([-1, 9, 10])
x.bitwise_or_(y)

x = torch.tensor([True, False, True])
y = torch.tensor([False, False, True])
x.bitwise_or_(y)

# torch.Tensor.bitwise_right_shift
input = torch.tensor([-2, -7, 31], dtype=torch.int8)
other = torch.tensor([1, 0, 3], dtype=torch.int8)
result = input.bitwise_right_shift(other)

input = torch.tensor([-2, -7, 31], dtype=torch.int8)
other = torch.tensor([1, 0, 3], dtype=torch.int8)
result = input.bitwise_right_shift(other=other)

# torch.Tensor.bitwise_right_shift_
input = torch.tensor([-2, -7, 31], dtype=torch.int8)
other = torch.tensor([1, 0, 3], dtype=torch.int8)
result = input.bitwise_right_shift_(other)

input = torch.tensor([-2, -7, 31], dtype=torch.int8)
other = torch.tensor([1, 0, 3], dtype=torch.int8)
result = input.bitwise_right_shift_(other=other)

# torch.Tensor.bitwise_xor
x = torch.tensor([1, 2, 3])
y = torch.tensor([-1, 9, 10])
result = x.bitwise_xor(y)

x = torch.tensor([True, False, True])
y = torch.tensor([False, False, True])
result = x.bitwise_xor(y)

# torch.Tensor.bitwise_xor_
x = torch.tensor([1, 2, 3])
y = torch.tensor([-1, 9, 10])
x.bitwise_xor_(y)

x = torch.tensor([True, False, True])
y = torch.tensor([False, False, True])
x.bitwise_xor_(y)

# torch.Tensor.bmm
a = torch.tensor([[[4., 5., 6.], [1., 2., 3.]]])
b = torch.tensor([[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]])
result = a.bmm(b)

result = torch.tensor([[[4., 5., 6.], [1., 2., 3.]]]).bmm(torch.tensor([[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]]))

# torch.Tensor.broadcast_to
x = torch.tensor([1, 2, 3])
result = x.broadcast_to((3, 3))

x = torch.tensor([1, 2, 3])
shape = [3, 3]
result = x.broadcast_to(size=shape)

# torch.Tensor.cauchy_
x = torch.randn([3, 4])
result = x.cauchy_()

x = torch.randn([3, 4])
result = x.cauchy_(1.0, 2.0)

# torch.Tensor.ceil
a = torch.tensor([1.1, 2.5, 3.6, 4.8])
result = a.ceil()

result = torch.tensor([0.1, -1.5, -2.3, 3.8]).ceil()

# torch.Tensor.ceil_
result = torch.tensor([-0.6341, -1.4208, -1.0900,  0.5826]).ceil_()

input = torch.tensor([-0.6341, -1.4208, -1.0900,  0.5826])
result = input.ceil_()

# torch.Tensor.cholesky
x = torch.tensor([[ 2.4112, -0.7486,  1.4551],
                [-0.7486,  1.3544,  0.1294],
                [ 1.4551,  0.1294,  1.6724]])
result = x.cholesky()

x = torch.tensor([[ 2.4112, -0.7486,  1.4551],
                [-0.7486,  1.3544,  0.1294],
                [ 1.4551,  0.1294,  1.6724]])
result = x.cholesky(True)

# torch.Tensor.cholesky_inverse
a = torch.tensor([[ 0.9967,  0.0000,  0.0000],
    [-0.6374,  0.6860,  0.0000],
    [ 1.5858, -1.0314,  2.6615]])
result = a.cholesky_inverse()

a = torch.tensor([[ 0.9967, -0.6374,  1.5858],
    [ 0.0000,  0.6860, -1.0314],
    [ 0.0000,  0.0000,  2.6615]])
result = a.cholesky_inverse(upper=True)

# torch.Tensor.chunk
x = torch.ones(2, 3)
result = x.chunk(2)

result = torch.ones(2, 3).chunk(chunks=2)

# torch.Tensor.clamp
a = torch.tensor([-1.7120,  0.1734, -0.0478, 0.8922])
result = a.clamp(-0.5, 0.5)

a = torch.tensor([-1.7120,  0.1734, -0.0478, 0.8922])
result = a.clamp(min=-0.2, max=0.5)

# torch.Tensor.clamp_
a = torch.tensor([-1.7120,  0.1734, -0.0478, 0.8922])
result = a.clamp_(-0.5, 0.5)

a = torch.tensor([-1.7120,  0.1734, -0.0478, 0.8922])
result = a.clamp_(min=-0.2, max=0.5)

# torch.Tensor.clip
x = torch.tensor([-1.7120,  0.1734, -0.0478, -0.0922])
result = x.clip(-0.5, 0.5)

x = torch.tensor([-1.7120,  0.1734, -0.0478, -0.0922])
min, max = -0.5, 0.5
result = x.clip(min, max)

# torch.Tensor.clip_
x = torch.tensor([-1.7120,  0.1734, -0.0478, -0.0922])
result = x.clip_(-0.5, 0.5)

x = torch.tensor([-1.7120,  0.1734, -0.0478, -0.0922])
min, max = -0.5, 0.5
result = x.clip_(min, max)

# torch.Tensor.clone
x = torch.tensor([1, 2, 3])
result = x.clone()

result = torch.tensor([1, 2, 3]).clone()

# torch.Tensor.coalesce
i = torch.tensor([[0, 1, 1],
                  [2, 0, 2]])
v = torch.tensor([3, 4, 5], dtype=torch.float32)
x = torch.sparse_coo_tensor(i, v, [2, 4])
result = x.coalesce()
result = result.to_dense()

# torch.Tensor.conj
src = torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j])
result = src.conj()

# torch.Tensor.contiguous
src = torch.tensor([1., 2., 3., 4., 5., 6.])
result = src.contiguous()

src = torch.tensor([1., 2., 3., 4., 5., 6.])
result = src.contiguous(memory_format=torch.contiguous_format)

# torch.Tensor.copy_
src = torch.tensor([1., 2., 3., 4., 5., 6.])
dst = torch.tensor([2., 2., 3., 4., 5., 6.])
result = src.copy_(dst)

src = torch.tensor([1., 2., 3., 4., 5., 6.])
dst = torch.tensor([2., 2., 3., 4., 5., 6.])
result = src.copy_(dst, non_blocking = False)

# torch.Tensor.copysign
a = torch.tensor([-1.2557, -0.0026, -0.5387,  0.4740, -0.9244])
result = a.copysign(1.)

a = torch.tensor([[ 0.7079,  0.2778, -1.0249,  0.5719],
[-0.0059, -0.2600, -0.4475, -1.3948],
[ 0.3667, -0.9567, -2.5757, -0.1751],
[ 0.2046, -0.0742,  0.2998, -0.1054]])
b = torch.tensor([ 0.2373,  0.3120,  0.3190, -1.1128])
result = a.copysign(b)

# torch.Tensor.copysign_
a = torch.tensor([-1.2557, -0.0026, -0.5387,  0.4740, -0.9244])
result = a.copysign_(1.)

a = torch.tensor([[ 0.7079,  0.2778, -1.0249,  0.5719],
[-0.0059, -0.2600, -0.4475, -1.3948],
[ 0.3667, -0.9567, -2.5757, -0.1751],
[ 0.2046, -0.0742,  0.2998, -0.1054]])
b = torch.tensor([ 0.2373,  0.3120,  0.3190, -1.1128])
result = a.copysign_(b)

# torch.Tensor.corrcoef
x = torch.tensor([[ 0.7308,  1.0060,  0.5270,  1.4516],
                [-0.1383,  1.5706,  0.4724,  0.4141],
                [ 0.1193,  0.2829,  0.9037,  0.3957],
                [-0.8202, -0.6474, -0.1631, -0.6543]])
result = x.corrcoef()

x = torch.tensor([[-0.1533,  2.3020, -0.1771,  0.5928],
                  [ 0.4338, -0.6537,  0.2296,  0.5946],
                  [-0.4932,  1.8386, -0.1039,  1.0440],
                  [ 0.1735, -0.8303, -0.3821, -0.4384],
                  [-0.1533,  2.3020, -0.1771,  0.5928],
                  [ 0.4338, -0.6537,  0.2296,  0.5946],
                  [-0.4932,  1.8386, -0.1039,  1.0440],
                  [ 0.1735, -0.8303, -0.3821, -0.4384]])
result = x.corrcoef()

# torch.Tensor.cos
result = torch.tensor([1.4309,  1.2706, -0.8562,  0.9796]).cos()

a = torch.tensor([ 1.4309,  1.2706, -0.8562,  0.9796])
result = a.cos()

# torch.Tensor.cos_
a = torch.tensor([ 1.4309,  1.2706, -0.8562,  0.9796])
a.cos_()

# torch.Tensor.cosh
result = torch.tensor([1.4309,  1.2706, -0.8562,  0.9796]).cosh()

a = torch.tensor([ 1.4309,  1.2706, -0.8562,  0.9796])
result = a.cosh()

# torch.Tensor.cosh_
a = torch.tensor([1.4309,  1.2706, -0.8562,  0.9796])
a.cosh_()

# torch.Tensor.cpu
a = torch.tensor([1,2,3])
result = a.cpu()

a = torch.tensor([1,2,3])
result = a.T.cpu()

# torch.Tensor.cross
x = torch.tensor([[1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0]])
y = torch.tensor([[1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0]])
result = x.cross(y)

x = torch.tensor([[1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0]])
y = torch.tensor([[1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0]])
result = x.cross(y, 1)

# torch.Tensor.cumprod
x = torch.tensor([[1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0]])
result = x.cumprod(0)

x = torch.tensor([[1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0]])
result = x.cumprod(dim=1)

# torch.Tensor.cumprod_
x = torch.tensor([[1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0]])
x.cumprod_(0)

x = torch.tensor([[1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0]])
x.cumprod_(dim=1)

# torch.Tensor.cumsum
x = torch.tensor([[1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0]])
result = x.cumsum(0)

x = torch.tensor([[1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0]])
result = x.cumsum(dim=1)

# torch.Tensor.cumsum_
x = torch.tensor([[1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0]])
x.cumsum_(0)

x = torch.tensor([[1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0]])
x.cumsum_(dim=1)

# torch.Tensor.data
x = torch.tensor([1.3192, 1.9915, 1.9674, 1.7151])
result = x.data

x = torch.tensor([1.3192, 1.9915, 1.9674, 1.7151])
x.data = torch.tensor([1., 1., 1., 1.])
result = x.data

# torch.Tensor.data_ptr
a = a = torch.tensor([[1, 2, 3], [1, 2, 3]])
result = a.data_ptr()

a = a = torch.tensor([[1., 2., 3.], [1., 2., 3.]])
result = a.data_ptr()

# torch.Tensor.deg2rad
x = torch.tensor([[1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0]])
result = x.deg2rad()

# torch.Tensor.dense_dim
i = torch.tensor([[0, 1, 1],
                  [2, 0, 2]])
v = torch.tensor([3, 4, 5], dtype=torch.float32)
x = torch.sparse_coo_tensor(i, v, [2, 4])
result = x.dense_dim()

# torch.Tensor.detach
x = torch.tensor([[1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0]], requires_grad=True)
result = x.detach()

result = torch.tensor([[1.0, 1.0, 1.0]], requires_grad=True).detach()

# torch.Tensor.detach_
x = torch.tensor([[1.0, 1.0, 1.0]], requires_grad=True)
x.detach_()

x = torch.tensor([[1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0]], requires_grad=True)
linear = torch.nn.Linear(3, 4, bias=False)
linear.weight.data.fill_(0.1)
y = linear(x)
y.detach_()

# torch.Tensor.device
src = torch.tensor([1., 2., 3., 4., 5., 6.]).to("cuda")
result = src.device

src = torch.tensor([1., 2., 3., 4., 5., 6.]).to("cpu")
result = src.device

# torch.Tensor.diag
a = torch.tensor([ 0.5950,-0.0872, 2.3298])
result = a.diag()

a = torch.tensor([ 0.5950,-0.0872, 2.3298])
result = a.diag(diagonal=1)

# torch.Tensor.diag_embed
x = torch.tensor([[ 0.7545889 , -0.25074545,  0.5929117 ], [-0.6097662 , -0.01753256,  0.619769  ]])
result = x.diag_embed()

x = torch.tensor([[ 0.7545889 , -0.25074545,  0.5929117 ], [-0.6097662 , -0.01753256,  0.619769  ]])
result = x.diag_embed(1)

# torch.Tensor.diagflat
a = torch.tensor([1,2,3])
result = a.diagflat()

a = torch.tensor([1,2,3])
result = a.diagflat(1)

# torch.Tensor.diagonal
x = torch.tensor([[ 0.7545889 , -0.25074545,  0.5929117 ], [-0.6097662 , -0.01753256,  0.619769  ]])
result = x.diagonal()

x = torch.tensor([[ 0.7545889 , -0.25074545,  0.5929117 ], [-0.6097662 , -0.01753256,  0.619769  ]])
result = x.diagonal(1)

# torch.Tensor.diff
x = torch.tensor([1, 3, 2])
result = x.diff()

x = torch.tensor([1, 3, 2])
b = torch.tensor([4, 5])
result = x.diff(append=b)

# torch.Tensor.digamma
result = torch.tensor([1, 0.5]).digamma()

a = torch.tensor([1, 0.5])
result = a.digamma()

# torch.Tensor.digamma_
a = torch.tensor([1, 0.5])
a.digamma_()

# torch.Tensor.dim
result = torch.tensor([1, 0.5]).dim()

a = torch.tensor([[1, 0.5]])
result = a.dim()

# torch.Tensor.dist
input = torch.tensor([-1.5393, -0.8675,  0.5916,  1.6321])
other = torch.tensor([ 0.0967, -1.0511,  0.6295,  0.8360])
result = input.dist(other, 2)

input = torch.tensor([-1.5393, -0.8675,  0.5916,  1.6321])
other = torch.tensor([ 0.0967, -1.0511,  0.6295,  0.8360])
result = input.dist(other, p=2.5)

# torch.Tensor.div
a = torch.tensor([ 0.5950,-0.0872, 2.3298, -0.2972])
result = a.div(torch.tensor([0.5]))

a = torch.tensor([ 0.5950,-0.0872, 2.3298, -0.2972])
result = a.div(0.5)

# torch.Tensor.div_
a = torch.tensor([ 0.5950,-0.0872, 2.3298, -0.2972])
a.div_(torch.tensor([0.5]))

a = torch.tensor([[ 0.5950,-0.0872], [2.3298, -0.2972]])
b = torch.tensor([0.1815, -1.0111])
a.div_(other=b)

# torch.Tensor.divide
a = torch.tensor([ 0.5950,-0.0872, 2.3298, -0.2972])
result = a.divide(torch.tensor([0.5]))

a = torch.tensor([ 0.5950,-0.0872, 2.3298, -0.2972])
result = a.divide(0.5)

# torch.Tensor.divide_
a = torch.tensor([ 0.5950,-0.0872, 2.3298, -0.2972])
a.divide_(torch.tensor([0.5]))

a = torch.tensor([[ 0.5950,-0.0872], [2.3298, -0.2972]])
b = torch.tensor([0.1815, -1.0111])
a.divide_(other=b)

# torch.Tensor.dot
result = torch.tensor([2, 3]).dot(torch.tensor([2, 1]))

x = torch.tensor([2, 3])
y = torch.tensor([2, 1])
result = x.dot(y)

# torch.Tensor.dtype
src = torch.tensor([1., 2., 3., 4., 5., 6.])
print(src.dtype)
result = torch.tensor([1,2,3], dtype=src.dtype )

# torch.Tensor.element_size
x = torch.tensor([1, 3, 2])
result = x.element_size()

x = torch.tensor([1, 3, 2], dtype=torch.int64)
result = x.element_size()

# torch.Tensor.eq
result = torch.tensor([[1, 2], [3, 4]]).eq(torch.tensor([[1, 1], [4, 4]]))

input = torch.tensor([[1, 2], [3, 4]])
other = torch.tensor([[1, 1], [4, 4]])
result = input.eq(other)

# torch.Tensor.erf
result = torch.tensor([0, -1., 10.]).erf()

a = torch.tensor([0, -1., 10.])
result = a.erf()

# torch.Tensor.erfinv
result = torch.tensor([0, 0.5]).erfinv()

a = torch.tensor([0, 0.5])
result = a.erfinv()

# torch.Tensor.erfinv_
result = torch.tensor([0, 0.5]).erfinv_()

a = torch.tensor([0, 0.5])
result = a.erfinv_()

# torch.Tensor.exp
result = torch.tensor([0., -2., 3.]).exp()

a = torch.tensor([-1., -2., 3.])
result = a.exp()

# torch.Tensor.exp_
result = torch.tensor([0., -2., 3.]).exp_()

a = torch.tensor([-1., -2., 3.])
result = a.exp_()

# torch.Tensor.expand
a = torch.tensor([1, 2, 3])
result = a.expand(3, 3)

result = torch.tensor([1, 2, 3]).expand(3, -1)

# torch.Tensor.expand_as
x = torch.tensor([[1], [2], [3]])
y = torch.randn(3, 4)
result = x.expand_as(y)

y = torch.randn(3, 4)
result = torch.tensor([[1], [2], [3]]).expand_as(y)

# torch.Tensor.expm1
a = torch.tensor([1., 2., -3., -4., 5.])
result = a.expm1()

a = torch.tensor([[1., 2., -3., -4., 5.], [1., 2., -3., -4., 5.]])
result = 2 * a.expm1()

# torch.Tensor.fill_
input = torch.rand([5, 9])
result = input.fill_(3)

input = torch.rand([5, 9])
result = input.fill_(value=3)

# torch.Tensor.flatten
x = torch.tensor([[[3.4742,  0.5466, -0.8008, -0.9079], [3.4742,  0.5466, -0.8008, -0.9079]]])
result = x.flatten()

x = torch.tensor([[[3.4742,  0.5466, -0.8008, -0.9079], [3.4742,  0.5466, -0.8008, -0.9079]]])
result = x.flatten(1)

# torch.Tensor.flip
x = torch.tensor([[0, 1],[2, 3]])
result = x.flip((0, 1))

x = torch.tensor([[0, 1],[2, 3]])
result = x.flip([0, 1])

# torch.Tensor.floor
input = torch.tensor([-0.8166,  1.5308, -0.2530, -0.2091])
result = input.floor()

result = torch.tensor([-0.8166,  1.5308, -0.2530, -0.2091]).floor()

# torch.Tensor.floor_
a = torch.tensor([1.1, 2.5, 3.6, 4.8])
result = a.floor_()

result = torch.tensor([0.1, -1.5, -2.3, 3.8])
result.floor_()

# torch.Tensor.floor_divide
a = torch.tensor([4.0, 3.0])
b = torch.tensor([2.0, 2.0])
result = a.floor_divide(b)

result = torch.tensor([4.0, 3.0]).floor_divide(torch.tensor([2.0, 2.0]))

# torch.Tensor.fmax
result = torch.tensor([[1, 2], [3, 4]]).fmax(torch.tensor([[1, 1], [4, 4]]))

input = torch.tensor([[1, 2], [3, 4]])
other = torch.tensor([[1, 1], [4, 4]])
result = input.fmax(other)

# torch.Tensor.fmin
result = torch.tensor([[1, 2], [3, 4]]).fmin(torch.tensor([[1, 1], [4, 4]]))

input = torch.tensor([[1, 2], [3, 4]])
other = torch.tensor([[1, 1], [4, 4]])
result = input.fmin(other)

# torch.Tensor.frac
result = torch.tensor([1, 2.5, -3.2]).frac()

a = torch.tensor([1, 2.5, -3.2])
result = a.frac()

# torch.Tensor.frac_
a = torch.tensor([1, 2.5, -3.2])
a.frac_()

# torch.Tensor.frexp
x = torch.arange(9.)
a,b = x.frexp()

# torch.Tensor.gather
a = torch.tensor([[1, 2], [3, 4]])
result = a.gather(1, torch.tensor([[0, 0], [1, 0]]))

result = torch.tensor([[1, 2], [3, 4]]).gather(1, torch.tensor([[0, 0], [1, 0]]))

# torch.Tensor.gcd
a = torch.tensor([5, 10, 15])
b = torch.tensor([3, 4, 5])
result = a.gcd(b)

a = torch.tensor([5, 10, 15])
b = torch.tensor([3, 4, 5])
result = a.gcd(other=b)

# torch.Tensor.gcd_
a = torch.tensor([5, 10, 15])
b = torch.tensor([3, 4, 5])
a.gcd_(b)

a = torch.tensor([5, 10, 15])
b = torch.tensor([3, 4, 5])
a.gcd_(other=b)

# torch.Tensor.ge
result = torch.tensor([[1, 2], [3, 4]]).ge(torch.tensor([[1, 1], [4, 4]]))

input = torch.tensor([[1, 2], [3, 4]])
other = torch.tensor([[1, 1], [4, 4]])
result = input.ge(other)

# torch.Tensor.geometric_
result = torch.tensor([-0.6341, -1.4208, -1.0900,  0.5826]).geometric_(0.5)

input = torch.tensor([-0.6341, -1.4208, -1.0900,  0.5826])
result = input.geometric_(0.5)

# torch.Tensor.get_device
x = torch.tensor([[1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0]]).cuda()
result = x.get_device()

result = None
x = torch.tensor([[1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0]]).cpu()
result = x.get_device()

# torch.Tensor.grad
src = torch.tensor([1., 2., 3., 4., 5., 6.])
result = src.grad

# torch.Tensor.greater
result = torch.tensor([[1, 2], [3, 4]]).greater(torch.tensor([[1, 1], [4, 4]]))

input = torch.tensor([[1, 2], [3, 4]])
other = torch.tensor([[1, 1], [4, 4]])
result = input.greater(other)

# torch.Tensor.greater_equal
result = torch.tensor([[1, 2], [3, 4]]).greater_equal(torch.tensor([[1, 1], [4, 4]]))

input = torch.tensor([[1, 2], [3, 4]])
other = torch.tensor([[1, 1], [4, 4]])
result = input.greater_equal(other)

# torch.Tensor.gt
result = torch.tensor([[1, 2], [3, 4]]).gt(torch.tensor([[1, 1], [4, 4]]))

input = torch.tensor([[1, 2], [3, 4]])
other = torch.tensor([[1, 1], [4, 4]])
result = input.gt(other)

# torch.Tensor.hypot_
a = torch.tensor([1., 2, 3])
b = torch.tensor([4., 5, 6])
result = a.hypot_(b)

a = torch.tensor([1., 2, 3])
b = torch.tensor([4., 5, 6])
result = a.hypot_(other=b)

# torch.Tensor.i0
a = torch.tensor([1.0000, 1.2661, 2.2796])
a.i0()

a = torch.tensor([1.0000, 1.2661, 2.2796]).i0()

# torch.Tensor.i0_
a = torch.tensor([1.0000, 1.2661, 2.2796])
a.i0_()

a = torch.tensor([1.0000, 1.2661, 2.2796]).i0_()

# torch.Tensor.index_add
x= torch.ones([5, 3])
t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
index = torch.tensor([0, 4, 2])
result = x.index_add(0, index, t)

x= torch.ones([5, 3])
t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
index = torch.tensor([0, 4, 2])
result = x.index_add(dim=0, index=index, source=t)

# torch.Tensor.index_add_
x = torch.ones([5, 3])
t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
index = torch.tensor([0, 4, 2])
x.index_add_(0, index, t)

x = torch.ones([5, 3])
t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
index = torch.tensor([0, 4, 2])
x.index_add_(dim=0, index=index, source=t)

# torch.Tensor.index_fill
x = torch.eye(2, 4)
indices = torch.tensor([0, 1])
value = -1
result = x.index_fill(0, indices, value)

indices = torch.tensor([0, 1])
value = -1
result = torch.eye(3, 4).index_fill(1, indices, value)

# torch.Tensor.index_fill_
x = torch.eye(2, 4)
indices = torch.tensor([0, 1])
value = -1
result = x.index_fill_(0, indices, value)

x = torch.eye(3, 4)
indices = torch.tensor([0, 1])
value = -1
result = x.index_fill_(1, indices, value)

# torch.Tensor.index_put
x = torch.ones([5, 3])
t = torch.tensor([1.], dtype=torch.float)
indices = [torch.tensor(i) for i in [[0, 0], [0, 1]]]
result = x.index_put(indices, t)

x = torch.ones([5, 3])
t = torch.tensor([1.], dtype=torch.float)
indices = [torch.tensor(i) for i in [[0, 0], [0, 1]]]
result = x.index_put(indices, values=t)

# torch.Tensor.index_put_
x = torch.ones([5, 3])
t = torch.tensor([1.], dtype=torch.float)
indices = [torch.tensor(i) for i in [[0, 0], [0, 1]]]
x.index_put_(indices, t)

x = torch.ones([5, 3])
t = torch.tensor([1.], dtype=torch.float)
indices = [torch.tensor(i) for i in [[0, 0], [0, 1]]]
x.index_put_(indices, values=t)

# torch.Tensor.index_select
x = torch.eye(2, 4)
indices = torch.tensor([0, 1])
result = x.index_select(0, indices)

indices = torch.tensor([0, 1])
result = torch.eye(3, 4).index_select(1, indices)

# torch.Tensor.indices
i = [[0, 1, 1], [2, 0, 2]]
v =  [3, 4, 5]
x = torch.sparse_coo_tensor(i, v, (2, 3)).coalesce()
result = x.indices()

# torch.Tensor.inverse
x = torch.tensor([[ 0.7308,  1.0060,  0.5270,  1.4516],
                [-0.1383,  1.5706,  0.4724,  0.4141],
                [ 0.1193,  0.2829,  0.9037,  0.3957],
                [-0.8202, -0.6474, -0.1631, -0.6543]])
result = x.inverse()

x = torch.tensor([[[[-0.1533,  2.3020, -0.1771,  0.5928],
                    [ 0.4338, -0.6537,  0.2296,  0.5946],
                    [-0.4932,  1.8386, -0.1039,  1.0440],
                    [ 0.1735, -0.8303, -0.3821, -0.4384]],
                    [[-0.1533,  2.3020, -0.1771,  0.5928],
                    [ 0.4338, -0.6537,  0.2296,  0.5946],
                    [-0.4932,  1.8386, -0.1039,  1.0440],
                    [ 0.1735, -0.8303, -0.3821, -0.4384]]]])
result = x.inverse()

# torch.Tensor.is_coalesced
i = torch.tensor([[0, 1, 1],
                  [2, 0, 2]])
v = torch.tensor([3, 4, 5])
x = torch.sparse_coo_tensor(i, v, [2, 4])
result = x.is_coalesced()

# torch.Tensor.is_complex
a = torch.tensor([[4, 9], [23, 2]])
result = a.is_complex()

result = torch.tensor([[4, 9], [23, 2]], dtype=torch.complex64).is_complex()

# torch.Tensor.is_contiguous
a = torch.tensor([[4, 9], [23, 2]])
result = a.is_contiguous()

result = torch.tensor([[4, 9], [23, 2]], dtype=torch.complex64).is_contiguous()

# torch.Tensor.is_cuda
x = torch.zeros(5, 3).cpu()
result = x.is_cuda

# torch.Tensor.is_floating_point
a = torch.tensor([[4, 9], [23, 2]], dtype=torch.int64)
result = a.is_floating_point()

a = torch.tensor([[4, 9], [23, 2]], dtype=torch.float64)
result = a.is_floating_point()

# torch.Tensor.is_leaf
src = torch.tensor([1., 2., 3., 4., 5., 6.])
result = src.is_leaf

# torch.Tensor.isclose
result = torch.tensor([10000., 1e-07]).isclose(torch.tensor([10000.1, 1e-08]))

result = torch.tensor([10000., 1e-08]).isclose(torch.tensor([10000.1, 1e-09]))

# torch.Tensor.isfinite
result = torch.tensor([1, float('inf'), 2, float('-inf'), float('nan')]).isfinite()

input = torch.tensor([1, float('inf'), 2, float('-inf'), float('nan')])
result = input.isfinite()

# torch.Tensor.isinf
result = torch.tensor([1, float('inf'), 2, float('-inf'), float('nan')]).isinf()

input = torch.tensor([1, float('inf'), 2, float('-inf'), float('nan')])
result = input.isinf()

# torch.Tensor.isnan
result = torch.tensor([1, float('inf'), 2, float('-inf'), float('nan')]).isnan()

input = torch.tensor([1, float('inf'), 2, float('-inf'), float('nan')])
result = input.isnan()

# torch.Tensor.isneginf
result = torch.tensor([1, float('inf'), 2, float('-inf'), float('nan')]).isneginf()

input = torch.tensor([1, float('inf'), 2, float('-inf'), float('nan')])
result = input.isneginf()

# torch.Tensor.isposinf
result = torch.tensor([1, float('inf'), 2, float('-inf'), float('nan')]).isposinf()

input = torch.tensor([1, float('inf'), 2, float('-inf'), float('nan')])
result = input.isposinf()

# torch.Tensor.isreal
x = torch.tensor([1, 1+1j, 2+0j])
result = x.isreal()

x = torch.tensor([-0., -2.1, 2.5])
result = x.isreal()

# torch.Tensor.istft
x = torch.tensor([[[ (5.975718021392822+0j)                  ,
   (5.975718021392822+0j)                  ,
   (5.341437339782715+0j)                  ,
   (5.404394626617432+0j)                  ,
   (5.404394626617432+0j)                  ],
 [ (0.0629572868347168+0j)                 ,
   0.0629572868347168j                     ,
  (-0.0629572868347168-0.6342806816101074j),
   (0.6342806816101074+0j)                 ,
   0.6342806816101074j                     ],
 [(-0.4979677200317383+0j)                 ,
   (0.4979677200317383+0j)                 ,
   (0.13631296157836914+0j)                ,
  (-0.19927024841308594+0j)                ,
   (0.19927024841308594+0j)                ]]])
result = x.istft(n_fft=4)

x = torch.tensor([[[ (5.975718021392822+0j)                  ,
   (5.975718021392822+0j)                  ,
   (5.341437339782715+0j)                  ,
   (5.404394626617432+0j)                  ,
   (5.404394626617432+0j)                  ],
 [ (0.0629572868347168+0j)                 ,
   0.0629572868347168j                     ,
  (-0.0629572868347168-0.6342806816101074j),
   (0.6342806816101074+0j)                 ,
   0.6342806816101074j                     ],
 [(-0.4979677200317383+0j)                 ,
   (0.4979677200317383+0j)                 ,
   (0.13631296157836914+0j)                ,
  (-0.19927024841308594+0j)                ,
   (0.19927024841308594+0j)                ]]])
result = x.istft(n_fft=4, center=False)

# torch.Tensor.item
a = torch.tensor([4])
result = a.item()

# torch.Tensor.itemsize
a = torch.tensor([-1])
result = a.itemsize

a = torch.tensor([-1, -2, 3])
result = a.itemsize

# torch.Tensor.lcm
a = torch.tensor([5, 10, 15])
b = torch.tensor([3, 4, 5])
result = a.lcm(b)

a = torch.tensor([5, 10, 15])
b = torch.tensor([3, 4, 5])
result = a.lcm(other=b)

# torch.Tensor.lcm_
a = torch.tensor([5, 10, 15])
b = torch.tensor([3, 4, 5])
a.lcm_(b)

a = torch.tensor([5, 10, 15])
b = torch.tensor([3, 4, 5])
a.lcm_(other=b)

# torch.Tensor.ldexp_
a = torch.tensor([1., 2., -3., -4., 5.])
b = torch.tensor([1., 2., -3., -4., 5.])
a.ldexp_(b)

a = torch.tensor([1., 2., -3., -4., 5.])
a.ldexp_(other=torch.tensor([1., 2., -3., -4., 5.]))

# torch.Tensor.le
result = torch.tensor([[1, 2], [3, 4]]).le(torch.tensor([[1, 1], [4, 4]]))

input = torch.tensor([[1, 2], [3, 4]])
other = torch.tensor([[1, 1], [4, 4]])
result = input.le(other)

# torch.Tensor.lerp
result = torch.tensor([1., 2., 3., 4.]).lerp(torch.tensor([10., 10., 10., 10.]), 0.5)

result = torch.tensor([1., 2., 3., 4.]).lerp(end=torch.tensor([10., 10., 10., 10.]), weight=0.5)

# torch.Tensor.lerp_
start = torch.tensor([1., 2., 3., 4.])
result = start.lerp_(torch.tensor([10., 10., 10., 10.]), 0.5)

start = torch.tensor([1., 2., 3., 4.])
result = start.lerp_(end=torch.tensor([10., 10., 10., 10.]), weight=0.5)

# torch.Tensor.less
result = torch.tensor([[1, 2], [3, 4]]).less(torch.tensor([[1, 1], [4, 4]]))

input = torch.tensor([[1, 2], [3, 4]])
other = torch.tensor([[1, 1], [4, 4]])
result = input.less(other)

# torch.Tensor.less_equal
result = torch.tensor([[1, 2], [3, 4]]).less_equal(torch.tensor([[1, 1], [4, 4]]))

input = torch.tensor([[1, 2], [3, 4]])
other = torch.tensor([[1, 1], [4, 4]])
result = input.less_equal(other)

# torch.Tensor.lgamma
input = torch.tensor([0.34, 1.5, 0.73])
result = input.lgamma()

result = torch.tensor([0.34, 1.5, 0.73]).lgamma()

# torch.Tensor.lgamma_
input = torch.tensor([0.34, 1.5, 0.73])
input.lgamma_()

input = torch.tensor([0.34, 1.5, 0.73]).lgamma_()

# torch.Tensor.log
input = torch.tensor([4.7767, 4.3234, 1.2156, 0.2411, 4.5739])
result = input.log()

result = torch.tensor([4.7767, 4.3234, 1.2156, 0.2411, 4.5739]).log()

# torch.Tensor.log10
input = torch.tensor([4.7767, 4.3234, 1.2156, 0.2411, 4.5739])
result = input.log10()

result = torch.tensor([4.7767, 4.3234, 1.2156, 0.2411, 4.5739]).log10()

# torch.Tensor.log10_
input = torch.tensor([4.7767, 4.3234, 1.2156, 0.2411, 4.5739])
input.log10_()

input = torch.tensor([4.7767, 4.3234, 1.2156, 0.2411, 4.5739]).log10_()

# torch.Tensor.log1p
input = torch.tensor([4.7767, 4.3234, 1.2156, 0.2411, 4.5739])
result = input.log1p()

result = torch.tensor([4.7767, 4.3234, 1.2156, 0.2411, 4.5739]).log1p()

# torch.Tensor.log1p_
input = torch.tensor([4.7767, 4.3234, 1.2156, 0.2411, 4.5739])
input.log1p_()

input = torch.tensor([4.7767, 4.3234, 1.2156, 0.2411, 4.5739]).log1p_()

# torch.Tensor.log2
input = torch.tensor([4.7767, 4.3234, 1.2156, 0.2411, 4.5739])
result = input.log2()

result = torch.tensor([4.7767, 4.3234, 1.2156, 0.2411, 4.5739]).log2()

# torch.Tensor.log2_
input = torch.tensor([4.7767, 4.3234, 1.2156, 0.2411, 4.5739])
input.log2_()

input = torch.tensor([4.7767, 4.3234, 1.2156, 0.2411, 4.5739]).log2_()

# torch.Tensor.log_
input = torch.tensor([4.7767, 4.3234, 1.2156, 0.2411, 4.5739])
input.log_()

input = torch.tensor([4.7767, 4.3234, 1.2156, 0.2411, 4.5739]).log_()

# torch.Tensor.logical_and
result = torch.tensor([True, False, True]).logical_and(torch.tensor([True, False, False]))

a = torch.tensor([0, 1, 10, 0], dtype=torch.int8)
b = torch.tensor([4, 0, 1, 0], dtype=torch.int8)
result = a.logical_and(b)

# torch.Tensor.logical_not
result = torch.tensor([True, False, True]).logical_not()

a = torch.tensor([0, 1, 10, 0], dtype=torch.int8)
result = a.logical_not()

# torch.Tensor.logical_or
result = torch.tensor([True, False, True]).logical_or(torch.tensor([True, False, False]))

a = torch.tensor([0, 1, 10, 0], dtype=torch.int8)
b = torch.tensor([4, 0, 1, 0], dtype=torch.int8)
result = a.logical_or(b)

# torch.Tensor.logical_xor
result = torch.tensor([True, False, True]).logical_xor(torch.tensor([True, False, False]))

a = torch.tensor([0, 1, 10, 0], dtype=torch.int8)
b = torch.tensor([4, 0, 1, 0], dtype=torch.int8)
result = a.logical_xor(b)

# torch.Tensor.logit
input = torch.tensor([0.2796, 0.9331, 0.6486, 0.1523, 0.6516])
result = input.logit(eps=1e-6)

input = torch.tensor([0.2796, 0.9331, 0.6486, 0.1523, 0.6516])
eps = 1e-6
result = input.logit(eps)

# torch.Tensor.logit_
input = torch.tensor([0.2796, 0.9331, 0.6486, 0.1523, 0.6516])
input.logit_(eps=1e-6)

input = torch.tensor([0.2796, 0.9331, 0.6486, 0.1523, 0.6516])
eps = 1e-6
input.logit_(eps)

# torch.Tensor.logsumexp
input = torch.tensor([1.4907, 1.0593, 1.5696])
result = input.logsumexp(0)

input = torch.tensor([[1.4907, 1.0593, 1.5696], [1.4907, 1.0593, 1.5696]])
result = input.logsumexp(1)

# torch.Tensor.lt
result = torch.tensor([[1, 2], [3, 4]]).lt(torch.tensor([[1, 1], [4, 4]]))

input = torch.tensor([[1, 2], [3, 4]])
other = torch.tensor([[1, 1], [4, 4]])
result = input.lt(other)

# torch.Tensor.lu
A = torch.tensor(
    [
        [
            [0.3591, -0.0479, -0.2174],
            [-0.6957, -1.4667, 1.4384],
            [0.0735, 0.1147, 0.0513],
        ],
        [
            [-1.2565, -2.1263, 0.8075],
            [-0.3665, -3.3540, -0.9417],
            [-0.1299, -0.0689, -0.6207],
        ],
    ]
)
A_LU, pivots = A.lu()

A = torch.tensor(
    [
        [
            [0.3591, -0.0479, -0.2174],
            [-0.6957, -1.4667, 1.4384],
            [0.0735, 0.1147, 0.0513],
        ],
        [
            [-1.2565, -2.1263, 0.8075],
            [-0.3665, -3.3540, -0.9417],
            [-0.1299, -0.0689, -0.6207],
        ],
    ]
)
A_LU, pivots, info = A.lu(get_infos=True)

# torch.Tensor.mT
x = torch.tensor([[0.+0.j, 1.+1.j],
        [2.+2.j, 3.+3.j]])
result = x.mT

x = torch.tensor([[1, 2, 3], [3, 4, 6]])
result = x.mT

# torch.Tensor.masked_fill
a = torch.Tensor([[1.0,0.2], [0.3,0.4]])
b = torch.Tensor([[1,0], [1,1]]) == 1
result = a.masked_fill(b, 2)

a = torch.Tensor([[1.0,0.2], [0.3,0.4]])
b = torch.Tensor([[1,0], [1,1]]) == 1
result = a.masked_fill(mask=b, value=2)

# torch.Tensor.masked_fill_
a = torch.Tensor([[1.0,0.2], [0.3,0.4]])
b = torch.Tensor([[1,0], [1,1]]) == 1
result = a.masked_fill_(b, 2)

a = torch.Tensor([[1.0,0.2], [0.3,0.4]])
b = torch.Tensor([[1,0], [1,1]]) == 1
result = a.masked_fill_(mask=b, value=2)

# torch.Tensor.masked_scatter
x = torch.tensor([0, 0, 0, 0, 0])
mask = torch.tensor([[0, 0, 0, 1, 1], [1, 1, 0, 1, 1]], dtype=torch.bool)
source = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
result = x.masked_scatter(mask, source)

x = torch.tensor([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
mask = torch.tensor([[0, 0, 0, 1, 1], [1, 1, 0, 1, 1]], dtype=torch.bool)
source = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
result = x.masked_scatter(mask, source)

# torch.Tensor.masked_scatter_
x = torch.tensor([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
mask = torch.tensor([[0, 0, 0, 1, 1], [1, 1, 0, 1, 1]], dtype=torch.bool)
source = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
result = x.masked_scatter_(mask, source)

# torch.Tensor.masked_select
x = torch.eye(2, 4)
mask = x > 0
result = x.masked_select(mask)

x = torch.ones(2, 4)
result = x.masked_select(x>0)

# torch.Tensor.matmul
x = torch.tensor([[4., 5., 6.], [1., 2., 3.], [4., 9., 10.]])
y = torch.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
result = x.matmul(y)

x = torch.tensor([[4., 5., 6.], [1., 2., 3.], [4., 9., 10.]])
y = torch.tensor([1., 2., 3.])
result = x.matmul(y)

# torch.Tensor.matrix_power
x = torch.tensor([[4., 5., 6.], [1., 2., 3.], [4., 9., 10.]])
result = x.matrix_power(2)

x = torch.tensor([[4., 5., 6.], [1., 2., 3.], [4., 9., 10.]])
result = x.matrix_power(-2)

# torch.Tensor.maximum
result = torch.tensor([[1, 2], [3, 4]]).maximum(torch.tensor([[1, 1], [4, 4]]))

input = torch.tensor([[1, 2], [3, 4]])
other = torch.tensor([[1, 1], [4, 4]])
result = input.maximum(other)

# torch.Tensor.mean
input = torch.tensor([1.4907, 1.0593, 1.5696])
result = input.mean()

input = torch.tensor([[1.4907, 1.0593, 1.5696], [1.4907, 1.0593, 1.5696]])
result = input.mean(1)

# torch.Tensor.minimum
result = torch.tensor([[1, 2], [3, 4]]).minimum(torch.tensor([[1, 1], [4, 4]]))

input = torch.tensor([[1, 2], [3, 4]])
other = torch.tensor([[1, 1], [4, 4]])
result = input.minimum(other)

# torch.Tensor.mm
a = torch.tensor([[1., 2.], [4., 5.]])
b = torch.tensor([[1., 3.], [3., 6.]])
result = a.mm(b)

a = torch.tensor([[1., 2.], [4., 5.]])
b = torch.tensor([[1., 3.], [3., 6.]])
result = a.mm(mat2=b)

# torch.Tensor.moveaxis
x = torch.arange(24)
x = torch.reshape(x, (1, 4, 6))
result = x.moveaxis(1, 0)

x = torch.arange(24)
x = torch.reshape(x, (1, 4, 6))
result = x.moveaxis((1, 0), (0, 1))

# torch.Tensor.msort
x = torch.tensor([[-1.3029,  0.4921, -0.7432],
                [ 2.6672, -0.0987,  0.0750],
                [ 0.1436, -1.0114,  1.3641]])
result = x.msort()

x = torch.tensor([[-1.3029,  0.4921, -0.7432],
                [ 2.6672, -0.0987,  0.0750],
                [ 0.1436, -1.0114,  1.3641]])
result = x.msort() * 3.

# torch.Tensor.mul
input = torch.tensor([0.2015, -0.4255, 2.6087])
other = torch.tensor([0.2015, -0.4255, 2.6087])
result = input.mul(other)

input = torch.tensor([0.2015, -0.4255,  2.6087])
other = torch.tensor([2, 6, 4])
result = input.mul(other=other)

# torch.Tensor.mul_
input = torch.tensor([0.2015, -0.4255, 2.6087])
other = torch.tensor([0.2015, -0.4255, 2.6087])
result = input.mul_(other)

input = torch.tensor([0.2015, -0.4255,  2.6087])
result = input.mul_(other=5.0)

# torch.Tensor.multinomial
torch.manual_seed(100)
weights = torch.tensor([0, 10, 3, 0], dtype=torch.float)
result = weights.multinomial(2)

torch.manual_seed(100)
weights = torch.tensor([0, 10, 3, 0], dtype=torch.float)
result = weights.multinomial(4, replacement=True)

# torch.Tensor.multiply
input = torch.tensor([0.2015, -0.4255, 2.6087])
other = torch.tensor([0.2015, -0.4255, 2.6087])
result = input.multiply(other)

input = torch.tensor([0.2015, -0.4255,  2.6087])
other = torch.tensor([2, 6, 4])
result = input.multiply(other=other)

# torch.Tensor.multiply_
input = torch.tensor([0.2015, -0.4255, 2.6087])
other = torch.tensor([0.2015, -0.4255, 2.6087])
result = input.multiply_(other)

input = torch.tensor([0.2015, -0.4255,  2.6087])
result = input.multiply_(other=5.0)

# torch.Tensor.mv
a = torch.tensor([[1., 2.], [4., 5.]])
b = torch.tensor([1., 3.])
result = a.mv(b)

a = torch.tensor([[1., 2.], [4., 5.]])
b = torch.tensor([1., 3.])
result = a.mv(vec=b)

# torch.Tensor.nan_to_num
input = torch.tensor([[1, 2], [3., float("nan")]])
result = input.nan_to_num()

input = torch.tensor([float('nan'), float('inf'), -float('inf'), 3.14])
result = input.nan_to_num(0., 1., -1.)

# torch.Tensor.nan_to_num_
input = torch.tensor([[1, 2], [3., float("nan")]])
input.nan_to_num_()

input = torch.tensor([float('nan'), float('inf'), -float('inf'), 3.14])
input.nan_to_num_(0., 1., -1.)

# torch.Tensor.narrow
x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
result = x.narrow(0, 0, 2)

x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
result = x.narrow(1, 1, 2)

# torch.Tensor.ndim
result = torch.tensor([1, 0.5]).ndim

a = torch.tensor([[1, 0.5]])
result = a.ndim

# torch.Tensor.ndimension
result = torch.tensor([1, 0.5]).ndimension()

a = torch.tensor([[1, 0.5]])
result = a.ndimension()

# torch.Tensor.ne
result = torch.tensor([[1, 2], [3, 4]]).ne(torch.tensor([[1, 1], [4, 4]]))

input = torch.tensor([[1, 2], [3, 4]])
other = torch.tensor([[1, 1], [4, 4]])
result = input.ne(other)

# torch.Tensor.neg
result = torch.tensor([-1, -2, 3]).neg()

a = torch.tensor([-1, -2, 3])
result = a.neg()

# torch.Tensor.neg_
a = torch.tensor([-1, -2, 3]).neg_()

a = torch.tensor([-1, -2, 3])
a.neg_()

# torch.Tensor.new_empty
x = torch.tensor([1., 2., 3.])
result = x.new_empty((1,))

x = torch.tensor([1., 2., 3.])
result = x.new_empty((1, 3), dtype=torch.float64)

# torch.Tensor.new_full
x = torch.tensor([1., 2., 3.])
result = x.new_full((1,), 3.1234)

x = torch.tensor([1., 2., 3.])
result = x.new_full((1, 3), 3.1234, dtype=torch.float64)

# torch.Tensor.new_ones
x = torch.tensor([1., 2., 3.])
result = x.new_ones((1,))

x = torch.tensor([1., 2., 3.])
result = x.new_ones((1, 3), dtype=torch.float64)

# torch.Tensor.new_zeros
x = torch.tensor([1., 2., 3.])
result = x.new_zeros((1,))

x = torch.tensor([1., 2., 3.])
result = x.new_zeros((1, 3), dtype=torch.float64)

# torch.Tensor.nextafter
input = torch.tensor([1.0, 2.0])
result = input.nextafter(torch.tensor([2.0, 1.0]))

input = torch.tensor([1.0, 2.0])
b = torch.tensor([2.0, 1.0])
result = input.nextafter(b)

# torch.Tensor.nonzero
result = torch.tensor([1, 1, 1, 0, 1]).nonzero()

result = torch.tensor([[0.6, 0.0, 0.0, 0.0],
                            [0.0, 0.4, 0.0, 0.0],
                            [0.0, 0.0, 1.2, 0.0],
                            [0.0, 0.0, 0.0,-0.4]]).nonzero()

# torch.Tensor.norm
input = torch.tensor([[[-12., -11., -10., -9. ],
                    [-8. , -7. , -6. , -5. ],
                    [-4. , -3. , -2. , -1. ]],
                    [[ 0. ,  1. ,  2. ,  3. ],
                    [ 4. ,  5. ,  6. ,  7. ],
                    [ 8. ,  9. ,  10.,  11.]]])
result = input.norm(p='fro')

input = torch.tensor([[-12., -11., -10., -9. ],
            [-8. , -7. , -6. , -5. ],
            [-4. , -3. , -2. , -1. ]])
result = input.norm(p='nuc')

# torch.Tensor.normal_
result = torch.Tensor([[1.,2.], [3.,4.]])
result.normal_()

result = torch.Tensor([[1.,2.], [3.,4.]])
result.normal_(0, 1)

# torch.Tensor.not_equal
result = torch.tensor([[1, 2], [3, 4]]).not_equal(torch.tensor([[1, 1], [4, 4]]))

input = torch.tensor([[1, 2], [3, 4]])
other = torch.tensor([[1, 1], [4, 4]])
result = input.not_equal(other)

# torch.Tensor.outer
x = torch.tensor([1., 2, 3])
y = torch.tensor([1., 2, 3, 4])
result = x.outer(y)

x = torch.tensor([1., 2, 3])
y = torch.tensor([1., 2, 3, 4])
result =x.outer(vec2=y)

# torch.Tensor.permute
x = torch.tensor([1., 2., 3., 4.])
result = x.permute(0)

x = torch.tensor([[1., 2.], [ 3., 4.]])
result = x.permute(0, 1)

# torch.Tensor.pin_memory
a = torch.Tensor([[1.,2.], [3.,4.]])
result = a.pin_memory()

# torch.Tensor.polygamma
x = torch.tensor([1.02, 2.21, 3.33])
result = x.polygamma(1)

x = torch.tensor([1.02, 2.21, 3.33, 4])
result = x.polygamma(1)

# torch.Tensor.polygamma_
x = torch.tensor([1.02, 2.21, 3.33])
x.polygamma_(1)

x = torch.tensor([1.02, 2.21, 3.33, 4])
x.polygamma_(n = 1)

# torch.Tensor.pow
a = torch.tensor([0.4331,  1.2475,  0.6834, -0.2791])
result = a.pow(2)

a = torch.tensor([0.4331,  1.2475,  0.6834, -0.2791])
b = torch.tensor([1, 2, 3, 4])
result = a.pow(b)

# torch.Tensor.prod
input = torch.tensor([1.4907, 1.0593, 1.5696])
result = input.prod()

input = torch.tensor([[1.4907, 1.0593, 1.5696], [1.4907, 1.0593, 1.5696]])
result = input.prod(1)

# torch.Tensor.quantile
x = torch.tensor([0., 1., 2., 3.],dtype=torch.float64)
result = x.quantile(0.6)

x = torch.tensor([0., 1., 2., 3.],dtype=torch.float64)
result = x.quantile(q=0.6)

# torch.Tensor.rad2deg
x = torch.tensor([[3.142, -3.142], [6.283, -6.283], [1.570, -1.570]])
result = x.rad2deg()

result = torch.tensor([[3.142, -3.142], [6.283, -6.283], [1.570, -1.570]]).rad2deg()

# torch.Tensor.random_
result = torch.tensor([-0.6341, -1.4208, -1.0900,  0.5826]).random_(0, 5)

input = torch.tensor([-0.6341, -1.4208, -1.0900,  0.5826])
result = input.random_(0, 5)

# torch.Tensor.ravel
a = torch.tensor([[4, 9], [23, 2]])
result = a.ravel()

result = torch.tensor([[4, 9], [23, 2]]).ravel()

# torch.Tensor.reciprocal
result = torch.tensor([-0.4595, -2.1219, -1.4314,  0.7298]).reciprocal()

a = torch.tensor([-0.4595, -2.1219, -1.4314,  0.7298])
result = a.reciprocal()

# torch.Tensor.reciprocal_
result = torch.tensor([-0.4595, -2.1219, -1.4314,  0.7298]).reciprocal_()

a = torch.tensor([-0.4595, -2.1219, -1.4314,  0.7298])
result = a.reciprocal_()

# torch.Tensor.register_hook
v = torch.tensor([0., 0., 0.], requires_grad=True)
h = v.register_hook(lambda grad: grad * 2)  # double the gradient
v.backward(torch.tensor([1., 2., 3.]))
result = torch.tensor(v.grad)

v = torch.tensor([0., 0., 0.], requires_grad=True)
h = v.register_hook(hook=lambda grad: grad * 2)  # double the gradient
v.backward(torch.tensor([1., 2., 3.]))
result = torch.tensor(v.grad)

# torch.Tensor.remainder
a = torch.tensor([-3., -2, -1, 1, 2, 3])
result = a.remainder(torch.tensor(2.))

a = torch.tensor([1, 2, 3, 4, 5])
result = a.remainder(torch.tensor(1.5))

# torch.Tensor.renorm
x = torch.tensor([[ 1.,  1.,  1.],
                    [ 2.,  2.,  2.],
                    [ 3.,  3.,  3.]])
result = x.renorm(1, 0, 5)

x = torch.tensor([[ 1.,  1.,  1.],
                    [ 2.,  2.,  2.],
                    [ 3.,  3.,  3.]])
result = x.renorm(p=1, dim=0, maxnorm=5)

# torch.Tensor.renorm_
x = torch.tensor([[ 1.,  1.,  1.],
                    [ 2.,  2.,  2.],
                    [ 3.,  3.,  3.]])
x.renorm_(1, 0, 5)

x = torch.tensor([[ 1.,  1.,  1.],
                    [ 2.,  2.,  2.],
                    [ 3.,  3.,  3.]])
x.renorm_(p=1, dim=0, maxnorm=5)

# torch.Tensor.repeat
x = torch.tensor([1, 2, 3])
result = x.repeat(4)

x = torch.tensor([1, 2, 3])
result = x.repeat(4, 2, 3)

# torch.Tensor.repeat_interleave
a = torch.tensor([[4, 9], [23, 2]])
result = a.repeat_interleave(3, 0)

a = torch.tensor([[4, 9], [23, 2]])
result = a.repeat_interleave(repeats=3, dim=1)

# torch.Tensor.requires_grad
data = torch.tensor([23.,32., 43.])
result = 1
if not data.requires_grad:
    result = 2

data = torch.tensor([23.,32., 43.])
result = data.requires_grad

# torch.Tensor.requires_grad_
a = torch.Tensor([[1.,2.], [3.,4.]])
result = a.requires_grad_()

a = torch.Tensor([[1.,2.], [3.,4.]])
result = a.requires_grad_(True)

# torch.Tensor.reshape
a = torch.arange(4.)
result = a.reshape(2, 2)

a = torch.arange(4.)
result = a.reshape((2, 2))

# torch.Tensor.roll
x = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
result = x.roll(1)

x = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
result = x.roll(1, 0)

# torch.Tensor.round
a = torch.tensor([[ 0.9254, -0.6213]])
result = a.round()

a = torch.tensor([[ 102003.9254, -12021.6213]])
result = a.round(decimals=1)

# torch.Tensor.round_
a = torch.tensor([[ 0.9254, -0.6213]])
result = a.round_()

a = torch.tensor([[ 102003.9254, -12021.6213]])
result = a.round_(decimals=1)

# torch.Tensor.rsqrt
result = torch.tensor([0.2970,  1.5420, 4]).rsqrt()

a = torch.tensor([0.2970,  1.5420, 4])
result = a.rsqrt()

# torch.Tensor.rsqrt_
result = torch.tensor([0.2970,  1.5420, 4]).rsqrt_()

result = torch.tensor([0.2970,  1.5420, 4])
result.rsqrt_()

# torch.Tensor.scatter
x = torch.arange(15).reshape([3, 5]).type(torch.float32)
index = torch.tensor([[0], [1], [2]])
result = x.scatter(1, index, 1.0)

x = torch.arange(15).reshape([3, 5]).type(torch.float32)
index = torch.tensor([[0], [1], [2]])
result = x.scatter(dim=1, index=index, value=1.0)

# torch.Tensor.scatter_
index = torch.tensor([[0],[1],[2]])
result = torch.zeros(3, 5).scatter_(1, index, 1.0)

index = torch.tensor([[0],[1],[2]])
result = torch.zeros(3, 5).scatter_(dim=1, index=index, value=1.0)

# torch.Tensor.scatter_add
src = torch.ones((1, 5))
index = torch.tensor([[0, 1, 2, 0, 0]])
result = torch.zeros(3, 5, dtype=src.dtype).scatter_add(0, index, src)

src = torch.ones((2, 5))
index = torch.tensor([[0, 1, 2, 0, 0], [0, 1, 2, 2, 2]])
result = torch.zeros(3, 5, dtype=src.dtype).scatter_add(dim=0, index=index, src=src)

# torch.Tensor.scatter_add_
src = torch.ones((1, 5))
index = torch.tensor([[0, 1, 2, 0, 0]])
result = torch.zeros(3, 5, dtype=src.dtype).scatter_add_(0, index, src)

src = torch.ones((2, 5))
index = torch.tensor([[0, 1, 2, 0, 0], [0, 1, 2, 2, 2]])
result = torch.zeros(3, 5, dtype=src.dtype).scatter_add_(0, index, src)

# torch.Tensor.scatter_reduce
src = torch.tensor([1., 2., 3., 4., 5., 6.])
index = torch.tensor([0, 1, 0, 1, 2, 1])
input = torch.tensor([1., 2., 3., 4.])
type = "sum"
result = input.scatter_reduce(0, index, src, reduce=type)

src = torch.tensor([1., 2., 3., 4., 5., 6.])
index = torch.tensor([0, 1, 0, 1, 2, 1])
input = torch.tensor([1., 2., 3., 4.])
re_type = "sum"
result = input.scatter_reduce(dim=0, index=index, src=src, reduce=re_type, include_self=False)

# torch.Tensor.sgn
a = torch.tensor([ 0.5950,-0.0872, 2.3298, -0.2972])
result = a.sgn()

a = torch.tensor([0.5950 + 0.3451j,-0.0872 - 0.3451j, 2.3298 + 0.3451j, -0.2972 + 0.3451j])
result = a.sgn()

# torch.Tensor.shape
src = torch.tensor([1., 2., 3., 4., 5., 6.])
result = src.shape

src = torch.empty((2, 0))
result = src.shape

# torch.Tensor.sigmoid
a = torch.Tensor([[1.,2.], [3.,4.]])
result = a.sigmoid()

# torch.Tensor.sigmoid_
a = torch.Tensor([[1.,2.], [3.,4.]])
a.sigmoid_()

# torch.Tensor.sign
result = torch.tensor([ 0.9213,  1.0887, -0.8858, -1.7683]).sign()

a = torch.tensor([ 0.9213,  1.0887, -0.8858, -1.7683])
result = a.sign()

# torch.Tensor.signbit
x = torch.tensor([-0., 1.1, -2.1, 0., 2.5], dtype=torch.float32)
result = x.signbit()

x = torch.tensor([-0., 1.1, -2.1, 0., 2.5], dtype=torch.float64)
result = x.signbit()

# torch.Tensor.sin
a = torch.Tensor([[1.,2.], [3.,4.]])
result = a.sin()

# torch.Tensor.sin_
a = torch.tensor([1.4309,  1.2706, -0.8562,  0.9796])
a.sin_()

# torch.Tensor.sinc
a = torch.tensor([ 0.5950,-0.0872, 0, -0.2972])
result = a.sinc()

result = torch.tensor([ 0.5950,-0.0872, 0, -0.2972]).sinc()

# torch.Tensor.sinc_
a = torch.tensor([ 0.5950,-0.0872, 0, -0.2972])
result = a.sinc_()

result = torch.tensor([ 0.5950,-0.0872, 0, -0.2972]).sinc_()

# torch.Tensor.sinh
a = torch.Tensor([[1.,2.], [3.,4.]])
result = a.sinh()

# torch.Tensor.sinh_
a = torch.tensor([1.4309,  1.2706, -0.8562,  0.9796])
a.sinh_()

# torch.Tensor.size
a = torch.tensor([ 0.5950,-0.0872, 0, -0.2972])
result = a.size()

a = torch.tensor([ 0.5950,-0.0872, 0, -0.2972])
result = a.size(dim=0)

# torch.Tensor.softmax
input = torch.tensor([[-1.2837, -0.0297,  0.0355],
    [ 0.9112, -1.7526, -0.4061]])
result = input.softmax(dim=0)

input = torch.tensor([[-1.2837, -0.0297,  0.0355],
    [ 0.9112, -1.7526, -0.4061]])
result = input.softmax(1)

# torch.Tensor.sparse_dim
i = torch.tensor([[0, 1, 1],
                  [2, 0, 2]])
v = torch.tensor([3, 4, 5], dtype=torch.float32)
x = torch.sparse_coo_tensor(i, v, [2, 4])
result = x.sparse_dim()

# torch.Tensor.sqrt
result = torch.tensor([0.2970,  1.5420, 4]).sqrt()

a = torch.tensor([0.2970,  1.5420, 4])
result = a.sqrt()

# torch.Tensor.sqrt_
result = torch.tensor([0.2970,  1.5420, 4]).sqrt_()

result = torch.tensor([0.2970,  1.5420, 4])
result.sqrt_()

# torch.Tensor.square
result = torch.tensor([0.2970,  1.5420, 4]).square()

a = torch.tensor([0.2970,  1.5420, 4])
result = a.square()

# torch.Tensor.squeeze
x = torch.zeros(2, 1, 2, 1, 2)
result =  x.squeeze()

result = torch.zeros(2, 1, 2, 1, 2).squeeze()

# torch.Tensor.std
input = torch.tensor([1.4907, 1.0593, 1.5696])
result = input.std(unbiased=False)

input = torch.tensor([[1.4907, 1.0593, 1.5696], [1.4907, 1.0593, 1.5696]])
result = input.std(unbiased=False)

# torch.Tensor.stride
a = torch.tensor([ 0.9254, -0.6213])
result = a.stride(dim=0)

a = torch.tensor([[ 0.9254, -0.6213], [ 0.9254, -0.6213]])
result = a.stride(dim=None)

# torch.Tensor.sub
x = torch.tensor([1, 2, 3])
result = x.sub(torch.tensor([1, 4, 6]))

x = torch.tensor([1, 2, 3])
result = x.sub(20)

# torch.Tensor.sub_
a = torch.tensor([0.5950, -0.0872, 2.3298, -0.2972])
b = torch.tensor([1., 2., 3., 4.])
a.sub_(b)

a = torch.tensor([ 0.5950,-0.0872, 2.3298, -0.2972])
b = torch.tensor([1., 2., 3., 4.])
a.sub_(other=b)

# torch.Tensor.sum
input = torch.tensor([1.4907, 1.0593, 1.5696])
result = input.sum()

input = torch.tensor([[1.4907, 1.0593, 1.5696], [1.4907, 1.0593, 1.5696]])
result = input.sum(1)

# torch.Tensor.swapaxes
x = torch.tensor([[[0,1],[2,3]],[[4,5],[6,7]]])
result = x.swapaxes(0, 1)

x = torch.tensor([[[0,1],[2,3]],[[4,5],[6,7]]])
result = x.swapaxes(axis0=0, axis1=1)

# torch.Tensor.swapdims
x = torch.tensor([[[0,1],[2,3]],[[4,5],[6,7]]])
result = x.swapdims(0, 1)

x = torch.tensor([[[0,1],[2,3]],[[4,5],[6,7]]])
result = x.swapdims(dim0=0, dim1=1)

# torch.Tensor.t
a = torch.Tensor([[1.,2.], [3.,4.]])
result = a.t()

# torch.Tensor.t_
a = torch.Tensor([[1.,2.], [3.,4.]])
result = a.t_()

# torch.Tensor.take_along_dim
x = torch.tensor([[1, 2, 3], [3, 4, 6]])
idx = torch.tensor([[0]])
result = x.take_along_dim(idx, 1)

x = torch.tensor([[1, 2, 3], [3, 4, 6]])
idx = torch.tensor([[0]])
result = x.take_along_dim(indices=idx, dim=0)

# torch.Tensor.tan
a = torch.Tensor([[1.,2.], [3.,4.]])
result = a.tan()

# torch.Tensor.tan_
a = torch.Tensor([[1.,2.], [3.,4.]])
a.tan_()

# torch.Tensor.tanh
a = torch.Tensor([[1.,2.], [3.,4.]])
result = a.tanh()

# torch.Tensor.tanh_
result = torch.Tensor([[1.,2.], [3.,4.]])
result.tanh_()

# torch.Tensor.tensor_split
a = torch.arange(8)
result = a.tensor_split(4)

a = torch.arange(7)
result = a.tensor_split(sections = 3)

# torch.Tensor.to
cpu = torch.device('cpu')
a =torch.ones(2, 3)
c = torch.ones(2, 3, dtype= torch.float64, device=cpu)
result = a.to(cpu, non_blocking=False, copy=False)

cpu = torch.device('cpu')
a =torch.ones(2, 3)
result = a.to('cpu')

# torch.Tensor.to_dense
i = torch.tensor([[0, 1, 1],
                  [2, 0, 2]])
v = torch.tensor([3, 4, 5], dtype=torch.float32)
x = torch.sparse_coo_tensor(i, v, [2, 4])
result = x.to_dense()

# torch.Tensor.tolist
a = torch.Tensor([[1.,2.], [3.,4.]])
result =torch.tensor(a.tolist())

# torch.Tensor.topk
a = torch.Tensor([[1.,2.], [3.,4.]])
result = a.topk(2)

a = torch.Tensor([[1.,2.], [3.,4.]])
result = a.topk(2, dim=0)

# torch.Tensor.trace
a = torch.Tensor([[1.,2.], [3.,4.]])
result = a.trace()

# torch.Tensor.transpose
a = torch.Tensor([[1.,2.], [3.,4.]])
result = a.transpose(dim0=0, dim1=1)

a = torch.Tensor([[1.,2.], [3.,4.]])
result = a.transpose(0, 1)

# torch.Tensor.tril
a = torch.tensor([[1.3192, 1.9915, 1.9674, 1.7151]])
result = a.tril()

a = torch.tensor([[1.3192, 1.9915, 1.9674, 1.7151]])
result = a.tril(1)

# torch.Tensor.tril_
x = torch.tensor([[-1.0813, -0.8619,  0.7105],
                [ 0.0935,  0.1380,  2.2112],
                [-0.3409, -0.9828,  0.0289]])
x.tril_()

x = torch.tensor([[-1.0813, -0.8619,  0.7105],
                [ 0.0935,  0.1380,  2.2112],
                [-0.3409, -0.9828,  0.0289]])
x.tril_(1)

# torch.Tensor.triu
a = torch.tensor([[1.3192, 1.9915, 1.9674, 1.7151]])
result = a.triu()

a = torch.tensor([[1.3192, 1.9915, 1.9674, 1.7151]])
result = a.triu(1)

# torch.Tensor.triu_
x = torch.tensor([[-1.0813, -0.8619,  0.7105],
                [ 0.0935,  0.1380,  2.2112],
                [-0.3409, -0.9828,  0.0289]])
x.triu_()

x = torch.tensor([[-1.0813, -0.8619,  0.7105],
                [ 0.0935,  0.1380,  2.2112],
                [-0.3409, -0.9828,  0.0289]])
x.triu_(1)

# torch.Tensor.true_divide
a = torch.tensor([4.67, 9.76 , 8.53])
b = torch.tensor([3.5, 3.90, 1.83])
result = a.true_divide(b)

a = torch.tensor([[4., 9., 8.]])
b = torch.tensor([2., 3., 4.])
result = a.true_divide(other=b)

# torch.Tensor.trunc
a = torch.Tensor([[1.,2.], [3.,4.]])
result = a.trunc()

# torch.Tensor.trunc_
a = torch.Tensor([[1.,2.], [3.,4.]])
a.trunc_()

# torch.Tensor.type_as
src = torch.tensor([1., 2., 3., 4., 5., 6.])
a = torch.tensor([1])
result = src.type_as(a)

src = torch.tensor([1., 2., 3., 4., 5., 6.])
a = torch.tensor([1])
result = src.type_as(other=a)

# torch.Tensor.unbind
a = torch.Tensor([[1.,2.], [3.,4.]])
result = a.unbind()

a = torch.Tensor([[1.,2.], [3.,4.]])
result = a.unbind(dim=0)

# torch.Tensor.unflatten
a = torch.tensor([[ 0.2180,  1.0558,  0.1608,  0.9245],
        [ 1.3794,  1.4090,  0.2514, -0.8818],
        [-0.4561,  0.5123,  1.7505, -0.4094]])
result = a.unflatten(-1, (2, 2))

a = torch.tensor([[ 0.2180,  1.0558,  0.1608,  0.9245],
        [ 1.3794,  1.4090,  0.2514, -0.8818],
        [-0.4561,  0.5123,  1.7505, -0.4094]])
result = a.unflatten(1, (2, 2))

# torch.Tensor.unfold
x = torch.arange(1., 8)
results = x.unfold(0, 2, 1)

x = torch.arange(1., 8)
results = x.unfold(0, 2, 2)

# torch.Tensor.uniform_
result = torch.Tensor([[1.,2.], [3.,4.]])
result.uniform_()

result = torch.Tensor([[1.,2.], [3.,4.]])
result.uniform_(0, to=1)

# torch.Tensor.unique_consecutive
x = torch.tensor([1, 1, 2, 2, 3, 1, 1, 2])
result = x.unique_consecutive()

x = torch.tensor([1, 1, 2, 2, 3, 1, 1, 2])
result, inverse_indices = x.unique_consecutive(return_inverse=True)

# torch.Tensor.unsqueeze
x = torch.zeros(2, 2, 2)
result = x.unsqueeze(0)

result = torch.zeros(2, 2, 1, 2).unsqueeze(3)

# torch.Tensor.values
i = [[0, 1, 1], [2, 0, 2]]
v =  [3, 4, 5]
x = torch.sparse_coo_tensor(i, v, (2, 3)).coalesce()
result = x.values()

# torch.Tensor.var
input = torch.tensor([1.4907, 1.0593, 1.5696])
result = input.var(unbiased=False)

input = torch.tensor([[1.4907, 1.0593, 1.5696], [1.4907, 1.0593, 1.5696]])
result = input.var(unbiased=False)

# torch.Tensor.view
a = torch.arange(4.)
result = a.view(2, 2)

a = torch.arange(4.)
result = a.view((2, 2))

# torch.Tensor.view_as
a = torch.ones([15])
b = torch.zeros([3, 5])
result = a.view_as(b)

a = torch.ones([15])
b = torch.zeros([3, 5])
result = a.view_as(other=b)

# torch.Tensor.zero_
result = torch.Tensor([[1.,2.], [3.,4.]])
result.zero_()

linear = torch.nn.Linear(5, 5)
result = linear.weight.data.zero_()

# torch.__version__
result = torch.__version__

# torch.__version__.split
result = torch.__version__.split()

result = torch.__version__.split(sep='234')

# torch.abs
result = torch.abs(torch.tensor([-1, -2, 3]))

a = torch.tensor([-1, -2, 3])
result = torch.abs(a)

# torch.abs_
a = torch.tensor([-1])
torch.abs_(a)

a = torch.tensor([-1, -2, 3])
torch.abs_(a)

# torch.acos
result = torch.acos(torch.tensor([0.34, -0.56, 0.73]))

a = torch.tensor([0.34, -0.56, 0.73])
result = torch.acos(a)

# torch.acosh
result = torch.acosh(torch.tensor([1.3192, 1.9915, 1.9674, 1.7151]))

a = torch.tensor([1.3192, 1.9915, 1.9674, 1.7151])
result = torch.acosh(a)

# torch.adaptive_avg_pool1d
x = torch.tensor([[[-1.3020, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
result = torch.adaptive_avg_pool1d(x, 5)

x = torch.tensor([[[-1.3020, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
result = torch.adaptive_avg_pool1d(input=x, output_size=5)

# torch.add
result = torch.add(torch.tensor([1, 2, 3]), torch.tensor([1, 4, 6]))

result = torch.add(torch.tensor([1, 2, 3]), 20)

# torch.addmm
x = torch.tensor([[1, 2], [4, 5]])
mat1 = torch.tensor([[1, 2], [4, 5]])
mat2 = torch.tensor([[1, 2], [4, 5]])
result = torch.addmm(x, mat1, mat2)

x = torch.tensor([[1., 2], [4, 5]])
mat1 = torch.tensor([[1., 2], [4, 5]])
mat2 = torch.tensor([[1., 2], [4, 5]])
result = torch.addmm(x, mat1, mat2, beta=0.6, alpha=0.7)

# torch.all
a = torch.rand(1, 2).bool()
result = torch.all(a)

a = torch.rand(3, 4)
result = torch.all(a)

# torch.amax
x = torch.tensor([[1, 2, 3], [3, 4, 6]])
result = torch.amax(x)

x = torch.tensor([[1, 2, 3], [3, 4, 6]])
result = torch.amax(x, dim=1)

# torch.amin
x = torch.tensor([[1, 2, 3], [3, 4, 6]])
result = torch.amin(x)

x = torch.tensor([[1, 2, 3], [3, 4, 6]])
result = torch.amin(x, dim=1)

# torch.amp.autocast
model = torch.nn.Linear(10, 5, device="cuda")
input = torch.randn(4, 10, device="cuda")

with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=False, cache_enabled=True):
    result = model(input)

model = torch.nn.Linear(10, 5, device="cuda")
input = torch.randn(4, 10, device="cuda")

with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=False, cache_enabled=True):
    result = model(input)

# torch.angle
result = torch.angle(torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j]))

x = torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j])
result = torch.angle(x) * 180 / 3.14159

# torch.any
a = torch.rand(1, 2).bool()
result = torch.any(a)

a = torch.rand(3, 4)
result = torch.any(a)

# torch.arange
result = torch.arange(5)

result = torch.arange(5.)

# torch.argmax
input = torch.tensor([[1.3398, 0.2663, -0.2686, 0.2450],
                    [-0.7401, -0.8805, -0.3402, -1.1936],
                    [0.4907, -1.3948, -1.0691, -0.3132],
                    [-1.6092, 0.5419, -0.2993, 0.3195]])
result = torch.argmax(input)

input = torch.tensor([[1.3398, 0.2663, -0.2686, 0.2450],
                    [-0.7401, -0.8805, -0.3402, -1.1936],
                    [0.4907, -1.3948, -1.0691, -0.3132],
                    [-1.6092, 0.5419, -0.2993, 0.3195]])
result = torch.argmax(input, dim = 1)

# torch.argmin
input = torch.tensor([[1.3398, 0.2663, -0.2686, 0.2450],
                    [-0.7401, -0.8805, -0.3402, -1.1936],
                    [0.4907, -1.3948, -1.0691, -0.3132],
                    [-1.6092, 0.5419, -0.2993, 0.3195]])
result = torch.argmin(input)

input = torch.tensor([[1.3398, 0.2663, -0.2686, 0.2450],
                    [-0.7401, -0.8805, -0.3402, -1.1936],
                    [0.4907, -1.3948, -1.0691, -0.3132],
                    [-1.6092, 0.5419, -0.2993, 0.3195]])
result = torch.argmin(input, dim = 1)

# torch.argsort
input = torch.tensor([[1.3398, 0.2663, -0.2686, 0.2450],
                    [-0.7401, -0.8805, -0.3402, -1.1936],
                    [0.4907, -1.3948, -1.0691, -0.3132],
                    [-1.6092, 0.5419, -0.2993, 0.3195]])
result = torch.argsort(input)

input = torch.tensor([[1.3398, 0.2663, -0.2686, 0.2450],
                    [-0.7401, -0.8805, -0.3402, -1.1936],
                    [0.4907, -1.3948, -1.0691, -0.3132],
                    [-1.6092, 0.5419, -0.2993, 0.3195]])
result = torch.argsort(input, dim = 1)

# torch.argwhere
input = torch.tensor([[1.0, 0.0, 0.0],
               [0.0, 2.0, 0.0],
               [0.0, 0.0, 3.0]])
result = torch.argwhere(input)

input = torch.tensor([0.0, 1.0, 0.0, 3.0])
result = torch.argwhere(input=input)

# torch.as_strided
x = torch.tensor([[0.0335, 0.1830, -0.1269],
[0.1897, -0.1422, -0.4940],
[-0.7674, -0.0134, -0.3733]])
result = torch.as_strided(x, (2, 2), (1, 2))

x = torch.tensor([[0.0335, 0.1830, -0.1269],
[0.1897, -0.1422, -0.4940],
[-0.7674, -0.0134, -0.3733]])
result = torch.as_strided(x, (2, 2), (1, 2), 0)

# torch.as_tensor
a = numpy.array([1, 2, 3])
result = torch.as_tensor(a)

result = torch.as_tensor(numpy.array([1, 2, 3]))

# torch.asarray
data = [[0, 1], [2, 3]]
result = torch.asarray(data)

data = [[0, 1], [2, 3]]
result = torch.asarray(data, dtype=torch.float64)

# torch.asin
result = torch.asin(torch.tensor([0.34, -0.56, 0.73]))

a = torch.tensor([0.34, -0.56, 0.73])
result = torch.asin(a)

# torch.asinh
result = torch.asinh(torch.tensor([0.1606, -1.4267, -1.0899, -1.0250]))

a = torch.tensor([0.1606, -1.4267, -1.0899, -1.0250])
result = torch.asinh(a)

# torch.atan
result = torch.atan(torch.tensor([0.2341, 0.2539, -0.6256, -0.6448]))

a = torch.tensor([0.2341, 0.2539, -0.6256, -0.6448])
result = torch.atan(a)

# torch.atan2
input = torch.tensor([0.9041, 0.0196, -0.3108, -2.4423])
other = torch.tensor([0.2341, 0.2539, -0.6256, -0.6448])
result = torch.atan2(input, other)

result = torch.atan2(torch.tensor([0.9041, 0.0196, -0.3108, -2.4423]), torch.tensor([0.2341, 0.2539, -0.6256, -0.6448]))

# torch.atanh
result = torch.atanh(torch.tensor([ -0.9385, 0.2968, -0.8591, -0.1871]))

a = torch.tensor([ -0.9385, 0.2968, -0.8591, -0.1871])
result = torch.atanh(a)

# torch.atleast_1d
result = torch.atleast_1d(torch.tensor(123, dtype=torch.int32))

y = torch.tensor([-1, -2, 3])
result = torch.atleast_1d((torch.tensor(123, dtype=torch.int32), y))

# torch.atleast_2d
result = torch.atleast_2d(torch.tensor(123, dtype=torch.int32))

y = torch.tensor([-1, -2, 3])
result = torch.atleast_2d((torch.tensor(123, dtype=torch.int32), y))

# torch.atleast_3d
result = torch.atleast_3d(torch.tensor(123, dtype=torch.int32))

y = torch.tensor([-1, -2, 3])
result = torch.atleast_3d((torch.tensor(123, dtype=torch.int32), y))

# torch.autocast
x = torch.tensor([[[-1.3020, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
with torch.autocast(device_type='cpu', enabled=False):
    result = x*x

x = torch.tensor([[[-1.3020, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
with torch.autocast(device_type='cpu'):
    result = x*x

# torch.autograd.Function
# Inherit from Function
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

# torch.baddbmm
a = torch.tensor([[[4., 5., 6.], [1., 2., 3.]]])
b = torch.tensor([[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]])
input = torch.tensor([[[1., 2., 3.], [4., 5., 6.]]])
result = torch.baddbmm(input, a, b)

a = torch.tensor([[[4., 5., 6.], [1., 2., 3.]]])
b = torch.tensor([[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]])
input = torch.tensor([[[1., 2., 3.], [4., 5., 6.]]])
result = torch.baddbmm(input, a, b, beta=3)

# torch.bfloat16
src = torch.tensor([1., 2., 3., 4., 5., 6.])
result = src.to(torch.bfloat16).to(torch.float)

result = torch.tensor([1., 2., 3., 4., 5., 6.]).to(torch.bfloat16).to(torch.float)

# torch.bincount
input = torch.tensor([4, 3, 6, 3, 4])
weights = torch.tensor([ 0.0000,  0.2500,  0.5000,  0.7500,  1.0000])
result = torch.bincount(input)

input = torch.tensor([4, 3, 6, 3, 4])
weights = torch.tensor([ 0.0000,  0.2500,  0.5000,  0.7500,  1.0000])
result = torch.bincount(input, weights)

# torch.bitwise_and
result = torch.bitwise_and(torch.tensor([-1, -2, 3], dtype=torch.int8), torch.tensor([1, 0, 3], dtype=torch.int8))

input = torch.tensor([-1, -2, 3], dtype=torch.int8)
other = torch.tensor([1, 0, 3], dtype=torch.int8)
result = torch.bitwise_and(input, other)

# torch.bitwise_left_shift
input = torch.tensor([-2, -7, 31], dtype=torch.int8)
other = torch.tensor([1, 0, 3], dtype=torch.int8)
result = torch.bitwise_left_shift(input, other)

input = torch.tensor([-2, -7, 31], dtype=torch.int8)
other = torch.tensor([1, 0, 3], dtype=torch.int8)
result = torch.bitwise_left_shift(input=input, other=other)

# torch.bitwise_not
result = torch.bitwise_not(torch.tensor([-1, -2, 3], dtype=torch.int8))

input = torch.tensor([-1, -2, 3], dtype=torch.int8)
result = torch.bitwise_not(input)

# torch.bitwise_or
result = torch.bitwise_or(torch.tensor([-1, -2, 3], dtype=torch.int8), torch.tensor([1, 0, 3], dtype=torch.int8))

input = torch.tensor([-1, -2, 3], dtype=torch.int8)
other = torch.tensor([1, 0, 3], dtype=torch.int8)
result = torch.bitwise_or(input, other)

# torch.bitwise_right_shift
input = torch.tensor([-2, -7, 31], dtype=torch.int8)
other = torch.tensor([1, 0, 3], dtype=torch.int8)
result = torch.bitwise_right_shift(input, other)

input = torch.tensor([-2, -7, 31], dtype=torch.int8)
other = torch.tensor([1, 0, 3], dtype=torch.int8)
result = torch.bitwise_right_shift(input=input, other=other)

# torch.bitwise_xor
result = torch.bitwise_xor(torch.tensor([-1, -2, 3], dtype=torch.int8), torch.tensor([1, 0, 3], dtype=torch.int8))

input = torch.tensor([-1, -2, 3], dtype=torch.int8)
other = torch.tensor([1, 0, 3], dtype=torch.int8)
result = torch.bitwise_xor(input, other)

# torch.blackman_window
result = torch.blackman_window(10)

result = torch.blackman_window(10, dtype=torch.float64)

# torch.block_diag
A = torch.tensor([[0, 1], [1, 0]])
B = torch.tensor([[3, 4, 5], [6, 7, 8]])
C = torch.tensor(7)
D = torch.tensor([1, 2, 3])
E = torch.tensor([[4], [5], [6]])
result = torch.block_diag(A, B, C, D, E)

A = torch.tensor([[4], [3], [2]])
B = torch.tensor([7, 6, 5])
C = torch.tensor(1)
result = torch.block_diag(torch.tensor([[4], [3], [2]]),
                        torch.tensor([7, 6, 5]),
                        torch.tensor(1))

# torch.bmm
a = torch.tensor([[[4., 5., 6.], [1., 2., 3.]]])
b = torch.tensor([[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]])
result = torch.bmm(a, b)

result = torch.bmm(torch.tensor([[[4., 5., 6.], [1., 2., 3.]]]), torch.tensor([[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]]))

# torch.bool
src = torch.tensor([1., 2., 3., 4., 5., 6.])
result = src.to(torch.bool)

result = torch.tensor([1., 2., 3., 4., 5., 6.]).to(torch.bool)

# torch.broadcast_shapes
x = (2,)
y = (3, 1)
result = torch.broadcast_shapes(x, y)

result = torch.broadcast_shapes((2,), (3, 1))

# torch.broadcast_tensors
x = torch.tensor([[0,1,2]])
y = torch.tensor([[0],[1]])
result = torch.broadcast_tensors(x, y)

y = torch.tensor([[0],[1]])
result = torch.broadcast_tensors(torch.tensor([[0,1,2]]), y)

# torch.broadcast_to
x = torch.tensor([1, 2, 3])
result = torch.broadcast_to(x, (3, 3))

x = torch.tensor([1, 2, 3])
shape = [3, 3]
result = torch.broadcast_to(input=x, size=shape)

# torch.bucketize
boundaries = torch.tensor([1, 3, 5, 7, 9])
v = torch.tensor([[3, 6, 9], [3, 6, 9]])
result = torch.bucketize(v, boundaries)

boundaries = torch.tensor([1, 3, 5, 7, 9])
v = torch.tensor([[3, 6, 9], [3, 6, 9]])
result = torch.bucketize(input=v, boundaries=boundaries, right=True)

# torch.cartesian_prod
a = torch.tensor([1, 2, 3])
b = torch.tensor([5, 6])
result = torch.cartesian_prod(a, b)

a = torch.tensor([1, 2, 3])
result = torch.cartesian_prod(a)

# torch.cat
x = torch.zeros(2, 3)
result = torch.cat((x, x, x))

x = torch.zeros(2, 3)
y = torch.zeros(2, 3)
result = torch.cat((x, y), 0)

# torch.cdist
x1 = torch.tensor([[ 1.6830,  0.0526],
    [-0.0696,  0.6366],
    [-1.0091,  1.3363]])
x2 = torch.tensor([[-0.0629,  0.2414],
    [-0.9701, -0.4455]])
result = torch.cdist(x1, x2)

x1 = torch.tensor([[ 1.6830,  0.0526],
    [-0.0696,  0.6366],
    [-1.0091,  1.3363]])
x2 = torch.tensor([[-0.0629,  0.2414],
    [-0.9701, -0.4455]])
result = torch.cdist(x1=x1, x2=x2, p=1.0, compute_mode='use_mm_for_euclid_dist_if_necessary')

# torch.ceil
result = torch.ceil(torch.tensor([-0.6341, -1.4208, -1.0900,  0.5826]))

input = torch.tensor([-0.6341, -1.4208, -1.0900,  0.5826])
result = torch.ceil(input)

# torch.cfloat
src = torch.tensor([1., 2., 3., 4., 5., 6.])
result = src.to(torch.cfloat)

result = torch.tensor([1., 2., 3., 4., 5., 6.]).to(torch.cfloat)

# torch.chunk
x = torch.ones(2, 3)
result = torch.chunk(x, 2)

result = torch.chunk(torch.ones(2, 3), chunks=2)

# torch.clamp
a = torch.tensor([-1.7120,  0.1734, -0.0478, 0.8922])
result = torch.clamp(a, -0.5, 0.5)

a = torch.tensor([-1.7120,  0.1734, -0.0478, 0.8922])
result = torch.clamp(a, min=-0.2, max=0.5)

# torch.clip
x = torch.tensor([-1.7120,  0.1734, -0.0478, -0.0922])
result = torch.clip(x, -0.5, 0.5)

x = torch.tensor([-1.7120,  0.1734, -0.0478, -0.0922])
min, max = -0.5, 0.5
result = torch.clip(x, min, max)

# torch.complex
real = torch.tensor([1, 2], dtype=torch.float32)
imag = torch.tensor([3, 4], dtype=torch.float32)
result = torch.complex(real, imag)

result = torch.complex(torch.tensor([1., 2]), torch.tensor([3., 4]))

# torch.complex128
src = torch.tensor([1., 2., 3., 4., 5., 6.])
result = src.to(torch.complex128)

result = torch.tensor([1., 2., 3., 4., 5., 6.]).to(torch.complex128)

# torch.complex64
src = torch.tensor([1., 2., 3., 4., 5., 6.])
result = src.to(torch.complex64)

result = torch.tensor([1., 2., 3., 4., 5., 6.]).to(torch.complex64)

# torch.concat
x = torch.zeros(2, 3)
result = torch.concat((x, x, x))

x = torch.zeros(2, 3)
y = torch.zeros(2, 3)
result = torch.concat((x, y), 0)

# torch.concatenate
x = torch.zeros(2, 3)
result = torch.concatenate((x, x, x))

x = torch.zeros(2, 3)
y = torch.zeros(2, 3)
result = torch.concatenate((x, y), 0)

# torch.conj
x = torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j])
result = torch.conj(x)

result = torch.conj(torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j]))

# torch.conv1d
x = torch.randn(33, 16, 30)
weight = torch.randn(20, 16, 5)
result = torch.conv1d(x, weight)

x = torch.randn(33, 16, 30)
weight = torch.randn(20, 16, 5)
bias = torch.randn(20)
result = torch.conv1d(x, weight, bias)

# torch.conv2d
x = torch.randn(33, 16, 30, 30)
weight = torch.randn(20, 16, 5, 5)
result = torch.conv2d(x, weight)

x = torch.randn(33, 16, 30, 30)
weight = torch.randn(20, 16, 5, 5)
bias = torch.randn(20)
result = torch.conv2d(x, weight, bias)

# torch.conv3d
x = torch.randn(33, 16, 30, 30, 30)
weight = torch.randn(20, 16, 5, 5, 5)
result = torch.conv3d(x, weight)

x = torch.randn(33, 16, 10, 10, 10)
weight = torch.randn(20, 16, 2, 2, 2)
bias = torch.randn(20)
result = torch.conv3d(x, weight, bias)

# torch.copysign
a = torch.tensor([1, 2, 3])
result = torch.copysign(a, -1, out=None)

a = torch.tensor([1, 2, 3, 4])
b = torch.tensor([-1, 2, -3, 4])
result = torch.copysign(a, b, out=None)

# torch.cos
result = torch.cos(torch.tensor([1.4309,  1.2706, -0.8562,  0.9796]))

a = torch.tensor([ 1.4309,  1.2706, -0.8562,  0.9796])
result = torch.cos(a)

# torch.cosh
result = torch.cosh(torch.tensor([1.4309,  1.2706, -0.8562,  0.9796]))

a = torch.tensor([ 1.4309,  1.2706, -0.8562,  0.9796])
result = torch.cosh(a)

# torch.cross
x = torch.tensor([[1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0]])
y = torch.tensor([[1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0]])
result = torch.cross(x, y)

x = torch.tensor([[1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0]])
y = torch.tensor([[1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0]])
result = torch.cross(x, y, 1)

# torch.cuda.BFloat16Tensor
result = torch.cuda.BFloat16Tensor(3, 5)

shape = [2, 2]
result = torch.cuda.BFloat16Tensor(*shape)

# torch.cuda.BoolTensor
result = torch.cuda.BoolTensor(2, 3)

shape = [2, 3]
result = torch.cuda.BoolTensor(*shape)

# torch.cuda.ByteTensor
result = torch.cuda.ByteTensor(2, 3)

shape = [2, 3]
result = torch.cuda.ByteTensor(*shape)

# torch.cuda.CharTensor
result = torch.cuda.CharTensor(2, 3)

shape = [2, 3]
result = torch.cuda.CharTensor(*shape)

# torch.cuda.DoubleTensor
result = torch.cuda.DoubleTensor(2, 3)

shape = [2, 3]
result = torch.cuda.DoubleTensor(*shape)

# torch.cuda.Event
result = torch.cuda.Event(enable_timing=True)

result = torch.cuda.Event(True, interprocess=False)

# torch.cuda.FloatTensor
result = torch.cuda.FloatTensor(2, 3)

shape = [2, 3]
result = torch.cuda.FloatTensor(*shape)

# torch.cuda.HalfTensor
result = torch.cuda.HalfTensor(2, 3)

shape = [2, 3]
result = torch.cuda.HalfTensor(*shape)

# torch.cuda.IntTensor
result = torch.cuda.IntTensor(2, 3)

shape = [2, 3]
result = torch.cuda.IntTensor(*shape)

# torch.cuda.LongTensor
result = torch.cuda.LongTensor(2, 3)

shape = [2, 3]
result = torch.cuda.LongTensor(*shape)

# torch.cuda.ShortTensor
result = torch.cuda.ShortTensor(2, 3)

shape = [2, 3]
result = torch.cuda.ShortTensor(*shape)

# torch.cuda.Stream
stream = torch.cuda.Stream()
result = stream.query()

stream = torch.cuda.Stream(priority=0)
result = stream.query()

# torch.cuda.StreamContext
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

# torch.cuda.amp.autocast
x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
with torch.cuda.amp.autocast(enabled=False):
    result = x*x

x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
with torch.cuda.amp.autocast():
    result = x*x

# torch.cuda.amp.autocast_mode.autocast
x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
with torch.cuda.amp.autocast_mode.autocast(enabled=False):
    result = x*x

x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
with torch.cuda.amp.autocast_mode.autocast():
    result = x*x

# torch.cuda.check_error
torch.cuda.check_error(0)

try:
    torch.cuda.check_error(1)
except RuntimeError as e:
    result1 = str(e)

try:
    torch.cuda.check_error(2)
except RuntimeError as e:
    result2 = str(e)

# torch.cuda.cudart
result = torch.cuda.cudart()

rt = torch.cuda.cudart()
result = rt.cudaMemGetInfo(0)

# torch.cuda.current_device
torch.cuda.set_device(0)
result = torch.cuda.current_device()

torch.cuda.set_device('cuda:0')
result = torch.cuda.current_device()

# torch.cuda.current_stream
result = torch.cuda.current_stream()

result = torch.cuda.current_stream(0)

# torch.cuda.device
with torch.cuda.device(0):
    result = torch.cuda.current_device()

with torch.cuda.device(device=1):
    result = torch.cuda.current_device()

# torch.cuda.device_count
result = torch.cuda.device_count()

# torch.cuda.empty_cache
result =torch.cuda.empty_cache()

# torch.cuda.get_device_capability
result = torch.cuda.get_device_capability(0)

result = torch.cuda.get_device_capability()

# torch.cuda.get_device_name
current_device = torch.cuda.current_device()
result = torch.cuda.get_device_name(current_device)

result = torch.cuda.get_device_name()

# torch.cuda.get_device_properties
result = torch.cuda.get_device_properties(torch.device(0))

result = torch.cuda.get_device_properties(device = "cuda:0")

# torch.cuda.get_rng_state
torch.cuda.get_rng_state()

torch.cuda.get_rng_state(device="cuda")

# torch.cuda.ipc_collect
result = torch.cuda.ipc_collect()

# torch.cuda.is_available
result = torch.cuda.is_available()

# torch.cuda.is_bf16_supported
result = torch.cuda.is_bf16_supported()

result = torch.cuda.is_bf16_supported(including_emulation=True)

# torch.cuda.is_current_stream_capturing
result = torch.cuda.is_current_stream_capturing()

# torch.cuda.is_initialized
torch.tensor([1], device='cuda:0')
result = torch.cuda.is_initialized()

x = torch.ones(2, 2).cuda()
result = torch.cuda.is_initialized()

# torch.cuda.manual_seed
torch.cuda.manual_seed(123)
result = torch.cuda.initial_seed()

torch.cuda.manual_seed(seed=123)
result = torch.cuda.initial_seed()

# torch.cuda.manual_seed_all
torch.cuda.manual_seed_all(123)
result = torch.cuda.initial_seed()

torch.cuda.manual_seed_all(seed=123)
result = torch.cuda.initial_seed()

# torch.cuda.max_memory_allocated
result = torch.cuda.max_memory_allocated()

t = torch.tensor([1,2,3]).cuda()
result = torch.cuda.max_memory_allocated()

# torch.cuda.max_memory_reserved
result = torch.cuda.max_memory_reserved()

t = torch.tensor([1,2,3]).cuda()
result = torch.cuda.max_memory_reserved()

# torch.cuda.mem_get_info
result = torch.cuda.mem_get_info()

t = torch.tensor([1,2,3]).cuda()
result = torch.cuda.mem_get_info()

# torch.cuda.memory_allocated
result = torch.cuda.memory_allocated()

t = torch.tensor([1,2,3]).cuda()
result = torch.cuda.memory_allocated()

# torch.cuda.memory_reserved
a = torch.tensor([1,2]).cuda()
result = torch.cuda.memory_reserved()

t = torch.tensor([1,2,3]).cuda()
result = torch.cuda.memory_reserved()

# torch.cuda.nvtx.range_pop
result = torch.cuda.nvtx.range_pop()

# torch.cuda.nvtx.range_push
result = torch.cuda.nvtx.range_push("msg")

result = torch.cuda.nvtx.range_push(msg="msg")

# torch.cuda.reset_peak_memory_stats
result = torch.cuda.reset_peak_memory_stats()

t = torch.tensor([1,2,3]).cuda()
result = torch.cuda.reset_peak_memory_stats(0)

# torch.cuda.set_device
torch.cuda.set_device("cuda:1")
result = torch.cuda.current_device()

torch.cuda.set_device(device=1)
result = torch.cuda.current_device()

# torch.cuda.set_rng_state
state = torch.cuda.get_rng_state(device='cuda')
rand1 = torch.rand([2,3], device='cuda')

torch.cuda.set_rng_state(state, device='cuda')
rand2 = torch.rand([2,3], device='cuda')

result = (rand2 - rand1)

# torch.cuda.set_stream
stream = torch.cuda.Stream(0)
result = torch.cuda.set_stream(stream)

stream = torch.cuda.Stream(torch.device("cuda:0"))
result = torch.cuda.set_stream(stream=stream)

# torch.cuda.stream
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

# torch.cuda.synchronize
result = torch.cuda.synchronize(0)

t = torch.tensor([1,2,3]).cuda()
result = torch.cuda.synchronize(device=0)

# torch.cumprod
x = torch.tensor([[1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0]])
result = torch.cumprod(x, 0)

x = torch.tensor([[1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0]])
result = torch.cumprod(x, 1, dtype=torch.float64)

# torch.cumsum
x = torch.tensor([[1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0]])
result = torch.cumsum(x, 0)

x = torch.tensor([[1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0]])
result = torch.cumsum(x, dim=1)

# torch.deg2rad
a = torch.tensor([[180.0, -180.0], [360.0, -360.0], [90.0, -90.0]])
result = torch.deg2rad(a)

result = torch.deg2rad(torch.tensor([[180.0, -180.0], [360.0, -360.0], [90.0, -90.0]]))

# torch.device
result = torch.device("{}".format("cpu"))

a = "cpu"
result = torch.device(a)

# torch.diag
x = torch.tensor([[-0.4264, 0.0255,-0.1064],
                [ 0.8795,-0.2429, 0.1374],
                [ 0.1029,-0.6482,-1.6300]])
result = torch.diag(x, 0)

x = torch.tensor([ 0.5950,-0.0872, 2.3298])
result = torch.diag(x)

# torch.diag_embed
x = torch.tensor([[ 0.7545889 , -0.25074545,  0.5929117 ], [-0.6097662 , -0.01753256,  0.619769  ]])
result = torch.diag_embed(x)

x = torch.tensor([[ 0.7545889 , -0.25074545,  0.5929117 ], [-0.6097662 , -0.01753256,  0.619769  ]])
result = torch.diag_embed(x, 1)

# torch.diagonal
x = torch.tensor([[ 0.7545889 , -0.25074545,  0.5929117 ], [-0.6097662 , -0.01753256,  0.619769  ]])
result = torch.diagonal(x)

x = torch.tensor([[ 0.7545889 , -0.25074545,  0.5929117 ], [-0.6097662 , -0.01753256,  0.619769  ]])
result = torch.diagonal(x, 1)

# torch.diff
x = torch.tensor([1, 3, 2])
result = torch.diff(x)

x = torch.tensor([1, 3, 2])
b = torch.tensor([4, 5])
result = torch.diff(x, append=b)

# torch.dist
input = torch.tensor([-1.5393, -0.8675,  0.5916,  1.6321])
other = torch.tensor([ 0.0967, -1.0511,  0.6295,  0.8360])
result = torch.dist(input, other, 2)

input = torch.tensor([-1.5393, -0.8675,  0.5916,  1.6321])
other = torch.tensor([ 0.0967, -1.0511,  0.6295,  0.8360])
result = torch.dist(input, other, p=2.5)

# torch.distributed.is_available
result = torch.distributed.is_available()

# torch.distributed.is_initialized
result=torch.distributed.is_initialized()

# torch.div
a = torch.tensor([ 0.5950,-0.0872, 2.3298, -0.2972])
result = torch.div(a, 0.5)

a = torch.tensor([[ 0.5950,-0.0872], [2.3298, -0.2972]])
b = torch.tensor([0.1815, -1.0111])
result = torch.div(a, b)

# torch.divide
a = torch.tensor([ 0.5950,-0.0872, 2.3298, -0.2972])
result = torch.divide(a, 0.5)

a = torch.tensor([[ 0.5950,-0.0872], [2.3298, -0.2972]])
b = torch.tensor([0.1815, -1.0111])
result = torch.divide(a, b)

# torch.dot
result = torch.dot(torch.tensor([2, 3]), torch.tensor([2, 1]))

x = torch.tensor([2, 3])
y = torch.tensor([2, 1])
result = torch.dot(x, y)

# torch.double
src = torch.tensor([1., 2., 3., 4., 5., 6.])
result = src.to(torch.double)

result = torch.tensor([1., 2., 3., 4., 5., 6.]).to(torch.double)

# torch.dtype
result = torch.tensor([1, 2, 3], dtype=torch.float16)

result = torch.tensor([1, 0, 3], dtype=torch.bool)

# torch.e
result = torch.e

# torch.einsum
x = torch.tensor([[1, 2, 3],[6, 2, 9], [1, 2, 3]])
result = torch.einsum('ii->i', x)

x = torch.tensor([[1, 2, 3], [6, 2, 9]])
result = torch.einsum('ij->ji', x)

# torch.empty
result = torch.empty(3)

result = torch.empty(3, 5)

# torch.empty_like
input = torch.empty((2,3), dtype=torch.int32)
result = torch.empty_like(input)

result = torch.empty_like(torch.empty(2, 3))

# torch.enable_grad
x = torch.tensor([1, 2, 3])
@torch.enable_grad()
def doubler(x):
    return x * 2
with torch.no_grad():
    result = doubler(x)

x = torch.tensor([1, 2, 3])
with torch.enable_grad():
    result = x ** 2

# torch.eq
result = torch.eq(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))

input = torch.tensor([[1, 2], [3, 4]])
other = torch.tensor([[1, 1], [4, 4]])
result = torch.eq(input, other)

# torch.exp
result = torch.exp(torch.tensor([0., -2., 3.]))

a = torch.tensor([-1., -2., 3.])
result = torch.exp(a)

# torch.expm1
result = torch.expm1(torch.tensor([0., -2., 3.]))

a = torch.tensor([-1., -2., 3.])
result = torch.expm1(a)

# torch.eye
result = torch.eye(3)

result = torch.eye(3, 5)

# torch.fft.fft
t = torch.arange(5)
result = torch.fft.fft(t)

t = torch.arange(5)
result = torch.fft.fft(input=t, n=2)

# torch.fft.fft2
t = torch.tensor([[3.+3.j, 2.+2.j, 3.+3.j], [2.+2.j, 2.+2.j, 3.+3.j]])
result = torch.fft.fft2(t)

t = torch.tensor([[3.+3.j, 2.+2.j, 3.+3.j], [2.+2.j, 2.+2.j, 3.+3.j]])
result = torch.fft.fft2(t, s=(2,3))

# torch.fft.fftfreq
result = torch.fft.fftfreq(5)

result = torch.fft.fftfreq(n=5, d=2)

# torch.fft.fftn
t = torch.tensor([[3.+3.j, 2.+2.j, 3.+3.j], [2.+2.j, 2.+2.j, 3.+3.j]])
result = torch.fft.fftn(t)

t = torch.tensor([[3.+3.j, 2.+2.j, 3.+3.j], [2.+2.j, 2.+2.j, 3.+3.j]])
result = torch.fft.fftn(t, s=(2,3))

# torch.fft.fftshift
t = torch.tensor([ 0.0000,  0.2500, -0.5000, -0.2500])
result = torch.fft.fftshift(t)

t = torch.tensor([ 0.0000,  0.2500, -0.5000, -0.2500])
result = torch.fft.fftshift(t, dim=(0,))

# torch.fft.hfft
t = torch.arange(5)
t = torch.linspace(0, 1, 5)
T = torch.fft.ifft(t)
result = torch.fft.hfft(T[:3], n=5)

t = torch.arange(5)
t = torch.linspace(0, 1, 5)
T = torch.fft.ifft(t)
result = torch.fft.hfft(T[:3])

# torch.fft.hfft2
t = torch.tensor([[3.+3.j, 2.+2.j, 3.+3.j], [2.+2.j, 2.+2.j, 3.+3.j]])
result = torch.fft.hfft2(t, s=(2, 5))

t = torch.tensor([[3.+3.j, 2.+2.j, 3.+3.j], [2.+2.j, 2.+2.j, 3.+3.j]])
result = torch.fft.hfft2(t)

# torch.fft.hfftn
t = torch.tensor([[3.+3.j, 2.+2.j, 3.+3.j], [2.+2.j, 2.+2.j, 3.+3.j]])
result = torch.fft.hfftn(t, s=(2, 5))

t = torch.tensor([[3.+3.j, 2.+2.j, 3.+3.j], [2.+2.j, 2.+2.j, 3.+3.j]])
result = torch.fft.hfftn(t)

# torch.fft.ifft
t = torch.arange(5)
result = torch.fft.ifft(t)

t = torch.arange(5)
result = torch.fft.ifft(input=t, n=2)

# torch.fft.ifft2
t = torch.tensor([[3.+3.j, 2.+2.j, 3.+3.j], [2.+2.j, 2.+2.j, 3.+3.j]])
result = torch.fft.ifft2(t)

t = torch.tensor([[3.+3.j, 2.+2.j, 3.+3.j], [2.+2.j, 2.+2.j, 3.+3.j]])
result = torch.fft.ifft2(t, s=(2,3))

# torch.fft.ifftn
t = torch.tensor([[3.+3.j, 2.+2.j, 3.+3.j], [2.+2.j, 2.+2.j, 3.+3.j]])
result = torch.fft.ifftn(t)

t = torch.tensor([[3.+3.j, 2.+2.j, 3.+3.j], [2.+2.j, 2.+2.j, 3.+3.j]])
result = torch.fft.ifftn(t, s=(2,3))

# torch.fft.ifftshift
t = torch.tensor([ 0.0000,  0.2500, -0.5000, -0.2500])
result = torch.fft.ifftshift(t)

t = torch.tensor([ 0.0000,  0.2500, -0.5000, -0.2500])
result = torch.fft.ifftshift(t, dim=(0,))

# torch.fft.ihfft
t = torch.arange(5)
result = torch.fft.ihfft(t)

t = torch.arange(5)
result = torch.fft.ihfft(input=t, n=2)

# torch.fft.ihfft2
t = torch.arange(20).reshape((4, 5)).type(torch.float64)
result = torch.fft.ihfft2(t, s=(2, 5))

t = torch.arange(20).reshape((4, 5)).type(torch.float64)
result = torch.fft.ihfft2(t)

# torch.fft.ihfftn
t = torch.arange(20).reshape((4, 5)).type(torch.float64)
result = torch.fft.ihfftn(t, s=(2, 5))

t = torch.arange(20).reshape((4, 5)).type(torch.float64)
result = torch.fft.ihfftn(t)

# torch.fft.irfft
t = torch.tensor([[3.+3.j, 2.+2.j, 3.+3.j], [2.+2.j, 2.+2.j, 3.+3.j]])
result = torch.fft.irfft(t)

t = torch.tensor([[3.+3.j, 2.+2.j, 3.+3.j], [2.+2.j, 2.+2.j, 3.+3.j]])
result = torch.fft.irfft(t, n=1)

# torch.fft.irfft2
t = torch.tensor([[3.+3.j, 2.+2.j, 3.+3.j], [2.+2.j, 2.+2.j, 3.+3.j]])
result = torch.fft.irfft2(t)

t = torch.tensor([[3.+3.j, 2.+2.j, 3.+3.j], [2.+2.j, 2.+2.j, 3.+3.j]])
result = torch.fft.irfft2(t, s=(2,3))

# torch.fft.irfftn
t = torch.Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
T = torch.fft.rfftn(t)
result = torch.fft.irfftn(T)

t = torch.Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
T = torch.fft.rfftn(t)
result = torch.fft.irfftn(T, norm="forward")

# torch.fft.rfft
t = torch.Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
result = torch.fft.rfft(t)

t = torch.Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
result = torch.fft.rfft(t, n=2)

# torch.fft.rfft2
t = torch.Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
result = torch.fft.rfft2(t)

t = torch.Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
result = torch.fft.rfft2(t, s=(4,2))

# torch.fft.rfftfreq
result = torch.fft.rfftfreq(5)

result = torch.fft.rfftfreq(n=5, d=2)

# torch.fft.rfftn
t = torch.Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
T = torch.fft.rfftn(t)
result = torch.fft.irfftn(T)

t = torch.Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
T = torch.fft.rfftn(t, s=(2,))
result = torch.fft.irfftn(T)

# torch.finfo
bits = torch.finfo(torch.float16).bits
min = torch.finfo(torch.float16).min
max = torch.finfo(torch.float16).max

x = torch.float32
bits = torch.finfo(x).bits
min = torch.finfo(x).min
max = torch.finfo(x).max

# torch.flatten
t = torch.tensor([[[1, 2],[3, 4]],[[5, 6],[7, 8]]])
result = torch.flatten(t)

t = torch.tensor([[[1, 2],[3, 4]],[[5, 6],[7, 8]]])
result = torch.flatten(t, start_dim=1)

# torch.flip
x = torch.tensor([[0, 1],[2, 3]])
result = torch.flip(x, (0, 1))

x = torch.tensor([[0, 1],[2, 3]])
result = torch.flip(x, [0, 1])

# torch.float16
src = torch.tensor([1., 2., 3., 4., 5., 6.])
result = src.to(torch.float16)

result = torch.tensor([1., 2., 3., 4., 5., 6.]).to(torch.float16)

# torch.float32
src = torch.tensor([1., 2., 3., 4., 5., 6.])
result = src.to(torch.float32)

result = torch.tensor([1., 2., 3., 4., 5., 6.]).to(torch.float32)

# torch.float64
src = torch.tensor([1., 2., 3., 4., 5., 6.])
result = src.to(torch.float64)

result = torch.tensor([1., 2., 3., 4., 5., 6.]).to(torch.float64)

# torch.float8_e4m3fn
src = torch.tensor([1., 2., 3., 4., 5., 6.])
result = src.to(torch.float8_e4m3fn).float()

result = torch.tensor([1., 2., 3., 4., 5., 6.]).to(torch.float8_e4m3fn).float()

# torch.floor
input = torch.tensor([-0.8166,  1.5308, -0.2530, -0.2091])
result = torch.floor(input)

result = torch.floor(torch.tensor([-0.8166,  1.5308, -0.2530, -0.2091]))

# torch.floor_divide
a = torch.tensor([4.0, 3.0])
b = torch.tensor([2.0, 2.0])
result = torch.floor_divide(a, b)

result = torch.floor_divide(torch.tensor([4.0, 3.0]), torch.tensor([2.0, 2.0]))

# torch.fmax
result = torch.fmax(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))

input = torch.tensor([[1, 2], [3, 4]])
other = torch.tensor([[1, 1], [4, 4]])
result = torch.fmax(input, other)

# torch.fmin
result = torch.fmin(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))

input = torch.tensor([[1, 2], [3, 4]])
other = torch.tensor([[1, 1], [4, 4]])
result = torch.fmin(input, other)

# torch.frac
result = torch.frac(torch.tensor([1, 2.5, -3.2]))

a = torch.tensor([1, 2.5, -3.2])
result = torch.frac(a)

# torch.from_numpy
a = numpy.array([1, 2, 3])
result = torch.from_numpy(a)

result = torch.from_numpy(numpy.array([1, 2, 3]))

# torch.full
input = torch.empty(2, 3)
result = torch.full(input.shape, 2.)

num = 5.
result = torch.full((2, 3), num)

# torch.full_like
input = torch.empty(2, 3)
result = torch.full_like(input, 2)

num = 5.
result = torch.full_like(torch.empty(2, 3), num)

# torch.functional.atleast_1d
result = torch.functional.atleast_1d(torch.tensor(123, dtype=torch.int32))

y = torch.tensor([-1, -2, 3])
result = torch.functional.atleast_1d((torch.tensor(123, dtype=torch.int32), y))

# torch.functional.atleast_2d
result = torch.functional.atleast_2d(torch.tensor(123, dtype=torch.int32))

y = torch.tensor([-1, -2, 3])
result = torch.functional.atleast_2d((torch.tensor(123, dtype=torch.int32), y))

# torch.functional.atleast_3d
result = torch.functional.atleast_3d(torch.tensor(123, dtype=torch.int32))

y = torch.tensor([-1, -2, 3])
result = torch.functional.atleast_3d((torch.tensor(123, dtype=torch.int32), y))

# torch.functional.broadcast_shapes
x = (2,)
y = (3, 1)
result = torch.functional.broadcast_shapes(x, y)

result = torch.functional.broadcast_shapes((2,), (3, 1))

# torch.functional.einsum
x = torch.tensor([[1, 2, 3],[6, 2, 9], [1, 2, 3]])
result = torch.functional.einsum('ii->i', x)

x = torch.tensor([[1, 2, 3], [6, 2, 9]])
result = torch.functional.einsum('ij->ji', x)

# torch.functional.meshgrid
x = torch.tensor([1, 2, 3])
y = torch.tensor([3, 4, 5])
grid_x, grid_y = torch.functional.meshgrid(x, y, indexing='ij')
result = grid_x

x = torch.tensor([1, 2, 3])
y = torch.tensor([3, 4, 5])
grid_x, grid_y = torch.functional.meshgrid(x, y, indexing='ij')
result = grid_y

# torch.functional.norm
input = torch.tensor([[[-12., -11., -10., -9. ],
                    [-8. , -7. , -6. , -5. ],
                    [-4. , -3. , -2. , -1. ]],
                    [[ 0. ,  1. ,  2. ,  3. ],
                    [ 4. ,  5. ,  6. ,  7. ],
                    [ 8. ,  9. ,  10.,  11.]]])
result = torch.functional.norm(input, p='fro')

input = torch.tensor([[-12., -11., -10., -9. ],
            [-8. , -7. , -6. , -5. ],
            [-4. , -3. , -2. , -1. ]])
result = torch.functional.norm(input, p='nuc')

# torch.functional.split
a = torch.arange(12).reshape(6, 2)
result = torch.functional.split(a, 2)

a = torch.arange(12).reshape(6, 2)
result = torch.functional.split(a, 2, dim=0)

# torch.functional.unique_consecutive
x = torch.tensor([1, 1, 2, 2, 3, 1, 1, 2])
result = torch.functional.unique_consecutive(x)

x = torch.tensor([1, 1, 2, 2, 3, 1, 1, 2])
result, inverse_indices = torch.functional.unique_consecutive(x, return_inverse=True)

# torch.gather
a = torch.tensor([[1, 2], [3, 4]])
result = torch.gather(a, 1, torch.tensor([[0, 0], [1, 0]]))

result = torch.gather(torch.tensor([[1, 2], [3, 4]]), 1, torch.tensor([[0, 0], [1, 0]]))

# torch.ge
result = torch.ge(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))

input = torch.tensor([[1, 2], [3, 4]])
other = torch.tensor([[1, 1], [4, 4]])
result = torch.ge(input, other)

# torch.ger
x = torch.tensor([1., 2, 3])
y = torch.tensor([1., 2, 3, 4])
result = torch.ger(x, y)

x = torch.tensor([1., 2, 3])
y = torch.tensor([1., 2, 3, 4])
result = torch.ger(input=x, vec2=y)

# torch.get_autocast_gpu_dtype
result_gpu = torch.get_autocast_gpu_dtype()

result_before = torch.get_autocast_gpu_dtype()
with torch.autocast("cuda",dtype=torch.bfloat16):
    result_inside = torch.get_autocast_gpu_dtype()
result_after = torch.get_autocast_gpu_dtype()

# torch.get_default_device
torch.set_default_device('cpu')
result = torch.get_default_device()

# if not set None, will cause test_vander error
torch.set_default_device(None)

torch.set_default_device(device=torch.device("cpu:1"))
result = torch.get_default_device()

# if not set None, will cause test_vander error
torch.set_default_device(None)

# torch.get_default_dtype
result = torch.get_default_dtype()

# torch.get_device
t = torch.tensor([1, 2, 3]).cuda()
result = torch.get_device(t)

# torch.get_device_module
result = torch.get_device_module("cuda")

result = torch.get_device_module(torch.device('cuda'))

# torch.get_rng_state
torch.get_rng_state()

# torch.greater
result = torch.greater(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))

input = torch.tensor([[1, 2], [3, 4]])
other = torch.tensor([[1, 1], [4, 4]])
result = torch.greater(input, other)

# torch.greater_equal
result = torch.greater_equal(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))

input = torch.tensor([[1, 2], [3, 4]])
other = torch.tensor([[1, 1], [4, 4]])
result = torch.greater_equal(input, other)

# torch.group_norm
x = torch.tensor([[[[-1.2392, -0.1310, -0.6679,  0.5476],
                    [ 1.1738, -1.7384, -0.7733,  0.3261],
                    [-0.0926, -1.0448, -1.2557, -1.5503],
                    [ 0.6402,  0.9072,  0.6780, -1.9885]],

                    [[ 0.0639, -1.1592,  1.4242, -0.4641],
                    [-0.1920,  0.1826,  1.9217, -0.4359],
                    [ 1.1926, -0.0247,  0.4744, -1.0216],
                    [-0.0360, -1.1656,  0.3661, -1.8147]]]])
result = torch.group_norm(x, 2)

x = torch.tensor([[[[-1.2392, -0.1310, -0.6679,  0.5476],
                    [ 1.1738, -1.7384, -0.7733,  0.3261],
                    [-0.0926, -1.0448, -1.2557, -1.5503],
                    [ 0.6402,  0.9072,  0.6780, -1.9885]],

                    [[ 0.0639, -1.1592,  1.4242, -0.4641],
                    [-0.1920,  0.1826,  1.9217, -0.4359],
                    [ 1.1926, -0.0247,  0.4744, -1.0216],
                    [-0.0360, -1.1656,  0.3661, -1.8147]]]])
result = torch.group_norm(x, 2)

# torch.gt
result = torch.gt(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))

input = torch.tensor([[1, 2], [3, 4]])
other = torch.tensor([[1, 1], [4, 4]])
result = torch.gt(input, other)

# torch.hamming_window
result = torch.hamming_window(10)

result = torch.hamming_window(10, dtype=torch.float64)

# torch.heaviside
input = torch.tensor([-1.5, 0, 2.0])
values = torch.tensor([0.5])
result = torch.heaviside(input, values)

input = torch.tensor([-1.5, 0, 2.0])
values = torch.tensor([0.5])
out = torch.tensor([-1.5, 0, 2.0])
torch.heaviside(input, values, out=out)

# torch.i0
result = torch.special.i0(torch.tensor([1.0000, 1.2661, 2.2796]))

a = torch.tensor([1.0000, 1.2661, 2.2796])
result = torch.special.i0(a)

# torch.index_add
x= torch.ones([5, 3])
t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
index = torch.tensor([0, 4, 2])
result = torch.index_add(x, 0, index, t)

x= torch.ones([5, 3])
t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
index = torch.tensor([0, 4, 2])
result = torch.index_add(input=x, dim=0, index=index, source=t)

# torch.index_fill
x = torch.eye(2, 4)
indices = torch.tensor([0, 1])
value = -1
result = torch.index_fill(x, 0, indices, value)

indices = torch.tensor([0, 1])
value = -1
result = torch.index_fill(torch.eye(3, 4), 1, indices, value)

# torch.index_put
x = torch.ones([5, 3])
t = torch.tensor([1.], dtype=torch.float)
indices = [torch.tensor(i) for i in [[0, 0], [0, 1]]]
result = torch.index_put(x, indices, t)

x = torch.ones([5, 3])
t = torch.tensor([1.], dtype=torch.float)
indices = [torch.tensor(i) for i in [[0, 0], [0, 1]]]
result = torch.index_put(x, indices, values=t)

# torch.index_select
x = torch.eye(2, 4)
indices = torch.tensor([0, 1])
result = torch.index_select(x, 0, indices)

indices = torch.tensor([0, 1])
result = torch.index_select(torch.eye(3, 4), 1, indices)

# torch.inf
result = torch.inf

# torch.int16
src = torch.tensor([1., 2., 3., 4., 5., 6.])
result = src.to(torch.int16)

result = torch.tensor([1., 2., 3., 4., 5., 6.]).to(torch.int16)

# torch.int32
src = torch.tensor([1., 2., 3., 4., 5., 6.])
result = src.to(torch.int32)

result = torch.tensor([1., 2., 3., 4., 5., 6.]).to(torch.int32)

# torch.int64
src = torch.tensor([1., 2., 3., 4., 5., 6.])
result = src.to(torch.int64)

result = torch.tensor([1., 2., 3., 4., 5., 6.]).to(torch.int64)

# torch.int8
src = torch.tensor([1., 2., 3., 4., 5., 6.])
result = src.to(torch.int8)

result = torch.tensor([1., 2., 3., 4., 5., 6.]).to(torch.int8)

# torch.inverse
x = torch.tensor([[ 0.7308,  1.0060,  0.5270,  1.4516],
                [-0.1383,  1.5706,  0.4724,  0.4141],
                [ 0.1193,  0.2829,  0.9037,  0.3957],
                [-0.8202, -0.6474, -0.1631, -0.6543]])
result = torch.inverse(x)

x = torch.tensor([[[[-0.1533,  2.3020, -0.1771,  0.5928],
                    [ 0.4338, -0.6537,  0.2296,  0.5946],
                    [-0.4932,  1.8386, -0.1039,  1.0440],
                    [ 0.1735, -0.8303, -0.3821, -0.4384]],
                    [[-0.1533,  2.3020, -0.1771,  0.5928],
                    [ 0.4338, -0.6537,  0.2296,  0.5946],
                    [-0.4932,  1.8386, -0.1039,  1.0440],
                    [ 0.1735, -0.8303, -0.3821, -0.4384]]]])
result = torch.inverse(x)

# torch.is_autocast_enabled
result = torch.is_autocast_enabled()

result_before = torch.is_autocast_enabled()
with torch.autocast(device_type='cuda', enabled=True):
    result_inside = torch.is_autocast_enabled()
result_after = torch.is_autocast_enabled()

# torch.is_complex
a = torch.tensor([[4, 9], [23, 2]])
result = torch.is_complex(a)

result = torch.is_complex(torch.tensor([[4, 9], [23, 2]], dtype=torch.complex64))

# torch.is_floating_point
a = torch.tensor([[4, 9], [23, 2]], dtype=torch.int64)
result = torch.is_floating_point(a)

a = torch.tensor([[4, 9], [23, 2]], dtype=torch.float64)
result = torch.is_floating_point(a)

# torch.is_grad_enabled
result=torch.is_grad_enabled()

# torch.is_tensor
a = torch.tensor([[4, 9], [23, 2]])
result = torch.is_tensor(a)

result = torch.is_tensor(torch.tensor([[4, 9], [23, 2]]))

# torch.isclose
result = torch.isclose(torch.tensor([10000., 1e-07]), torch.tensor([10000.1, 1e-08]))

result = torch.isclose(torch.tensor([10000., 1e-08]), torch.tensor([10000.1, 1e-09]))

# torch.isfinite
result = torch.isfinite(torch.tensor([1, float('inf'), 2, float('-inf'), float('nan')]))

input = torch.tensor([1, float('inf'), 2, float('-inf'), float('nan')])
result = torch.isfinite(input)

# torch.isin
result = torch.isin(torch.tensor([[1, 2], [3, 4]]), torch.tensor([2, 3]))

result = torch.isin(torch.tensor([[1, 2], [3, 4]]), torch.tensor([2, 3]), assume_unique=True)

# torch.isinf
result = torch.isinf(torch.tensor([1, float('inf'), 2, float('-inf'), float('nan')]))

input = torch.tensor([1, float('inf'), 2, float('-inf'), float('nan')])
result = torch.isinf(input)

# torch.isnan
result = torch.isnan(torch.tensor([1, float('inf'), 2, float('-inf'), float('nan')]))

input = torch.tensor([1, float('inf'), 2, float('-inf'), float('nan')])
result = torch.isnan(input)

# torch.layer_norm
input = torch.tensor([[[ 1.1524,  0.4714,  0.2857],
 [-1.2533, -0.9829, -1.0981],
 [ 0.1507, -1.1431, -2.0361]],

[[ 0.1024, -0.4482,  0.4137],
 [ 0.9385,  0.4565,  0.7702],
 [ 0.4135, -0.2587,  0.0482]]])
data = torch.tensor([1., 1., 1.])
result = torch.layer_norm(input, [3], data, data)

input = torch.tensor([[[ 1.1524,  0.4714,  0.2857],
 [-1.2533, -0.9829, -1.0981],
 [ 0.1507, -1.1431, -2.0361]],

[[ 0.1024, -0.4482,  0.4137],
 [ 0.9385,  0.4565,  0.7702],
 [ 0.4135, -0.2587,  0.0482]]])
data = torch.tensor([1., 1., 1.])
result = torch.layer_norm(input=input, normalized_shape=[3], weight=data, bias=data)

# torch.le
result = torch.le(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))

input = torch.tensor([[1, 2], [3, 4]])
other = torch.tensor([[1, 1], [4, 4]])
result = torch.le(input, other)

# torch.lerp
result = torch.lerp(torch.tensor([1., 2., 3., 4.]), torch.tensor([10., 10., 10., 10.]), 0.5)

result = torch.lerp(input=torch.tensor([1., 2., 3., 4.]), end=torch.tensor([10., 10., 10., 10.]), weight=0.5)

# torch.less
result = torch.less(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))

input = torch.tensor([[1, 2], [3, 4]])
other = torch.tensor([[1, 1], [4, 4]])
result = torch.less(input, other)

# torch.less_equal
result = torch.less_equal(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))

input = torch.tensor([[1, 2], [3, 4]])
other = torch.tensor([[1, 1], [4, 4]])
result = torch.less_equal(input, other)

# torch.linalg.matmul
x = torch.tensor([[4., 5., 6.], [1., 2., 3.], [4., 9., 10.]])
y = torch.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
result = torch.linalg.matmul(x, y)

x = torch.tensor([[4., 5., 6.], [1., 2., 3.], [4., 9., 10.]])
y = torch.tensor([1., 2., 3.])
result = torch.linalg.matmul(x, y)

# torch.linalg.norm
y = torch.tensor([[3, 4, 6], [5, 3, 4], [1, 2, 1.]])
result = torch.linalg.norm(y, dim=-1, keepdim=True)

x = torch.tensor([[10, 2, 3], [3, 10, 5], [5, 6, 12.]])
result = torch.linalg.norm(input=x, ord=float('inf'), dim=-1)

# torch.linalg.solve
x = torch.tensor([[3.0, 1],[1, 2]])
y = torch.tensor([9.0, 8])
result = torch.linalg.solve(x, y)

x = torch.tensor([[3.0, 1],[1, 2]])
y = torch.tensor([9.0, 8])
result = torch.linalg.solve(A=x, B=y)

# torch.linalg.vector_norm
x = torch.tensor([[0.02773777, 0.93004224, 0.06911496],
        [0.24831591, 0.45733623, 0.07717843],
        [0.48016702, 0.14235102, 0.42620817]])
result = torch.linalg.vector_norm(x)

x = torch.tensor([[0.02773777, 0.93004224, 0.06911496],
        [0.24831591, 0.45733623, 0.07717843],
        [0.48016702, 0.14235102, 0.42620817]])
result = torch.linalg.vector_norm(x=x, ord=2)

# torch.linspace
result = torch.linspace(3, 10, 5)

result = torch.linspace(-10., 10., 5)

# torch.log
input = torch.tensor([4.7767, 4.3234, 1.2156, 0.2411, 4.5739])
result = torch.log(input)

result = torch.log(torch.tensor([4.7767, 4.3234, 1.2156, 0.2411, 4.5739]))

# torch.log10
input = torch.tensor([4.7767, 4.3234, 1.2156, 0.2411, 4.5739])
result = torch.log10(input)

result = torch.log10(torch.tensor([4.7767, 4.3234, 1.2156, 0.2411, 4.5739]))

# torch.log1p
input = torch.tensor([4.7767, 4.3234, 1.2156, 0.2411, 4.5739])
result = torch.log1p(input)

result = torch.log1p(torch.tensor([4.7767, 4.3234, 1.2156, 0.2411, 4.5739]))

# torch.log2
input = torch.tensor([4.7767, 4.3234, 1.2156, 0.2411, 4.5739])
result = torch.log2(input)

result = torch.log2(torch.tensor([4.7767, 4.3234, 1.2156, 0.2411, 4.5739]))

# torch.logical_and
result = torch.logical_and(torch.tensor([True, False, True]), torch.tensor([True, False, False]))

a = torch.tensor([0, 1, 10, 0], dtype=torch.int8)
b = torch.tensor([4, 0, 1, 0], dtype=torch.int8)
result = torch.logical_and(a, b)

# torch.logical_not
result = torch.logical_not(torch.tensor([True, False, True]))

a = torch.tensor([0, 1, 10, 0], dtype=torch.int8)
result = torch.logical_not(a)

# torch.logical_or
result = torch.logical_or(torch.tensor([True, False, True]), torch.tensor([True, False, False]))

a = torch.tensor([0, 1, 10, 0], dtype=torch.int8)
b = torch.tensor([4, 0, 1, 0], dtype=torch.int8)
result = torch.logical_or(a, b)

# torch.logical_xor
result = torch.logical_xor(torch.tensor([True, False, True]), torch.tensor([True, False, False]))

a = torch.tensor([0, 1, 10, 0], dtype=torch.int8)
b = torch.tensor([4, 0, 1, 0], dtype=torch.int8)
result = torch.logical_xor(a, b)

# torch.logsumexp
input = torch.tensor([1.4907, 1.0593, 1.5696])
result = torch.logsumexp(input, 0)

input = torch.tensor([[1.4907, 1.0593, 1.5696], [1.4907, 1.0593, 1.5696]])
result = torch.logsumexp(input, 1)

# torch.long
src = torch.tensor([1., 2., 3., 4., 5., 6.])
result = src.to(torch.long)

result = torch.tensor([1., 2., 3., 4., 5., 6.]).to(torch.long)

# torch.lt
result = torch.lt(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))

input = torch.tensor([[1, 2], [3, 4]])
other = torch.tensor([[1, 1], [4, 4]])
result = torch.lt(input, other)

# torch.manual_seed
torch.manual_seed(100)
result = torch.initial_seed()

torch.manual_seed(seed=100)
result = torch.initial_seed()

# torch.masked_select
x = torch.eye(2, 4)
mask = x > 0
result = torch.masked_select(x, mask)

x = torch.ones(2, 4)
result = torch.masked_select(x, x>0)

# torch.matmul
x = torch.tensor([[4., 5., 6.], [1., 2., 3.], [4., 9., 10.]])
y = torch.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
result = torch.matmul(x, y)

x = torch.tensor([[4., 5., 6.], [1., 2., 3.], [4., 9., 10.]])
y = torch.tensor([1., 2., 3.])
result = torch.matmul(x, y)

# torch.maximum
result = torch.maximum(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))

input = torch.tensor([[1, 2], [3, 4]])
other = torch.tensor([[1, 1], [4, 4]])
result = torch.maximum(input, other)

# torch.mean
input = torch.tensor([1.4907, 1.0593, 1.5696])
result = torch.mean(input)

input = torch.tensor([[1.4907, 1.0593, 1.5696], [1.4907, 1.0593, 1.5696]])
result = torch.mean(input, 1)

# torch.meshgrid
x = torch.tensor([1, 2, 3])
y = torch.tensor([3, 4, 5])
grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
result = grid_x

x = torch.tensor([1, 2, 3])
y = torch.tensor([3, 4, 5])
grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
result = grid_y

# torch.minimum
result = torch.minimum(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))

input = torch.tensor([[1, 2], [3, 4]])
other = torch.tensor([[1, 1], [4, 4]])
result = torch.minimum(input, other)

# torch.mm
a = torch.tensor([[1., 2.], [4., 5.]])
b = torch.tensor([[1., 3.], [3., 6.]])
result = torch.mm(a, b)

a = torch.tensor([[1., 2.], [4., 5.]])
b = torch.tensor([[1., 3.], [3., 6.]])
result = torch.mm(input=a, mat2=b)

# torch.msort
x = torch.tensor([[-1.3029,  0.4921, -0.7432],
                [ 2.6672, -0.0987,  0.0750],
                [ 0.1436, -1.0114,  1.3641]])
result = torch.msort(x)

x = torch.tensor([[-1.3029,  0.4921, -0.7432],
                [ 2.6672, -0.0987,  0.0750],
                [ 0.1436, -1.0114,  1.3641]])
result = torch.msort(input = x)

# torch.mul
input = torch.tensor([0.2015, -0.4255, 2.6087])
other = torch.tensor([0.2015, -0.4255, 2.6087])
result = torch.mul(input, other)

input = torch.tensor([0.2015, -0.4255,  2.6087])
other = torch.tensor([2, 6, 4])
result = torch.mul(input, other)

# torch.multinomial
torch.manual_seed(100)
weights = torch.tensor([0, 10, 3, 0], dtype=torch.float)
result = torch.multinomial(weights, 2)

torch.manual_seed(100)
weights = torch.tensor([0, 10, 3, 0], dtype=torch.float)
result = torch.multinomial(weights, 4, replacement=True)

# torch.multiply
input = torch.tensor([0.2015, -0.4255, 2.6087])
other = torch.tensor([0.2015, -0.4255, 2.6087])
result = torch.multiply(input, other)

input = torch.tensor([0.2015, -0.4255,  2.6087])
other = torch.tensor([2, 6, 4])
result = torch.multiply(input, other)

# torch.nan
result = torch.nan

# torch.narrow
x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
result = torch.narrow(x, 0, 0, 2)

x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
result = torch.narrow(x, 1, 1, 2)

# torch.ne
result = torch.ne(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))

input = torch.tensor([[1, 2], [3, 4]])
other = torch.tensor([[1, 1], [4, 4]])
result = torch.ne(input, other)

# torch.newaxis
result = torch.newaxis

# torch.nextafter
input = torch.tensor([1.0, 2.0, 3.0])
other = torch.tensor([1.1, 2.1, 3.1])
result = torch.nextafter(input, other)

a = torch.tensor([0.0, 1.0, 2.0])
b = torch.tensor([0.5, 1.5, 2.5])
result = torch.nextafter(a, b)

# torch.nn.AdaptiveAvgPool1d
x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
model = nn.AdaptiveAvgPool1d(5)
result = model(x)

x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
model = nn.AdaptiveAvgPool1d(output_size=5)
result = model(x)

# torch.nn.AdaptiveAvgPool2d
x = torch.tensor([[[[ 0.9785,  1.2013,  2.4873, -1.1891],
                    [-0.0832, -0.5456, -0.5009,  1.5103],
                    [-1.2860,  1.0287, -1.3902,  0.4627],
                    [-0.0502, -1.3924, -0.3327,  0.1678]]]])
model = nn.AdaptiveAvgPool2d(5)
result = model(x)

x = torch.tensor([[[[ 0.9785,  1.2013,  2.4873, -1.1891],
                    [-0.0832, -0.5456, -0.5009,  1.5103],
                    [-1.2860,  1.0287, -1.3902,  0.4627],
                    [-0.0502, -1.3924, -0.3327,  0.1678]]]])
model = nn.AdaptiveAvgPool2d(output_size=(2, 2))
result = model(x)

# torch.nn.AdaptiveAvgPool3d
x = torch.tensor([[[[[-1.1494, -1.3829],
                    [ 0.4995, -1.3094]],
                    [[ 1.0015,  1.4919],
                    [-1.5187,  0.0235]]]]])
model = nn.AdaptiveAvgPool3d(1)
result = model(x)

x = torch.tensor([[[[[-1.1494, -1.3829],
                    [ 0.4995, -1.3094]],
                    [[ 1.0015,  1.4919],
                    [-1.5187,  0.0235]]]]])
model = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
result = model(x)

# torch.nn.AdaptiveLogSoftmaxWithLoss
input = torch.tensor([[ 0.9368637 , -0.0361056 , -0.98917043,  0.06605113,  1.5254455 ],
                    [-1.0518035 , -1.0024613 ,  0.18699688, -0.35807893,  0.25628588],
                    [-0.900478  , -0.41495147,  0.84707606, -1.7883497 ,  1.3243382 ]])
target = torch.tensor([1, 1, 1])
asfm = nn.AdaptiveLogSoftmaxWithLoss(5, 4, [2])
out, loss = asfm(input,target)

input = torch.tensor([[ 0.9368637 , -0.0361056 , -0.98917043,  0.06605113,  1.5254455 ],
                    [-1.0518035 , -1.0024613 ,  0.18699688, -0.35807893,  0.25628588],
                    [-0.900478  , -0.41495147,  0.84707606, -1.7883497 ,  1.3243382 ]])
target = torch.tensor([1, 1, 1])
asfm = nn.AdaptiveLogSoftmaxWithLoss(5, 4, [3], div_value=2.0)
out, loss = asfm(input,target)

# torch.nn.AdaptiveMaxPool1d
x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
model = nn.AdaptiveMaxPool1d(5)
result = model(x)

x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
model = nn.AdaptiveMaxPool1d(output_size=5)
result = model(x)

# torch.nn.AdaptiveMaxPool2d
x = torch.tensor([[[[ 0.9785,  1.2013,  2.4873, -1.1891],
                    [-0.0832, -0.5456, -0.5009,  1.5103],
                    [-1.2860,  1.0287, -1.3902,  0.4627],
                    [-0.0502, -1.3924, -0.3327,  0.1678]]]])
model = nn.AdaptiveMaxPool2d(5)
result = model(x)

x = torch.tensor([[[[ 0.9785,  1.2013,  2.4873, -1.1891],
                    [-0.0832, -0.5456, -0.5009,  1.5103],
                    [-1.2860,  1.0287, -1.3902,  0.4627],
                    [-0.0502, -1.3924, -0.3327,  0.1678]]]])
model = nn.AdaptiveMaxPool2d(output_size=(2, 2))
result = model(x)

# torch.nn.AdaptiveMaxPool3d
x = torch.tensor([[[[[-1.1494, -1.3829],
                    [ 0.4995, -1.3094]],
                    [[ 1.0015,  1.4919],
                    [-1.5187,  0.0235]]]]])
model = nn.AdaptiveMaxPool3d(1)
result = model(x)

x = torch.tensor([[[[[-1.1494, -1.3829],
                    [ 0.4995, -1.3094]],
                    [[ 1.0015,  1.4919],
                    [-1.5187,  0.0235]]]]])
model = nn.AdaptiveMaxPool3d(output_size=(1, 1, 1))
result = model(x)

# torch.nn.ChannelShuffle
x = torch.tensor([[[[0.]],[[0.10000000]],[[0.20000000]],[[0.30000001]],[[0.40000001]],[[0.50000000]]]])
model = nn.ChannelShuffle(3)
result = model(x)

x = torch.tensor([[[[0.]],[[0.10000000]],[[0.20000000]],[[0.30000001]],[[0.40000001]],[[0.50000000]]]])
model = nn.ChannelShuffle(groups=2)
result = model(x)

# torch.nn.CircularPad1d
x = torch.tensor([[[-1.3328, -0.4948],
                    [ 0.8689,  1.1423]],
                    [[-0.2671, -1.0868],
                    [ 1.3011,  1.0469]]])
model = nn.CircularPad1d(1)
result = model(x)
padding = model.padding

x = torch.tensor([[[-1.3328, -0.4948],
                    [ 0.8689,  1.1423]],
                    [[-0.2671, -1.0868],
                    [ 1.3011,  1.0469]]])
model = nn.CircularPad1d((1, 1,))
result = model(x)
padding = model.padding

# torch.nn.CircularPad2d
x = torch.tensor([[[[-1.3328, -0.4948],
                    [ 0.8689,  1.1423]],
                    [[-0.2671, -1.0868],
                    [ 1.3011,  1.0469]]]])
model = nn.CircularPad2d(1)
result = model(x)
padding = model.padding

x = torch.tensor([[[[-1.3328, -0.4948],
                    [ 0.8689,  1.1423]],
                    [[-0.2671, -1.0868],
                    [ 1.3011,  1.0469]]]])
model = nn.CircularPad2d((1, 1, 1, 1))
result = model(x)
padding = model.padding

# torch.nn.CircularPad3d
x = torch.tensor([[[[[-1.3328, -0.4948],
                    [ 0.8689,  1.1423]],
                    [[-0.2671, -1.0868],
                    [ 1.3011,  1.0469]]]]])
model = nn.CircularPad3d(1)
result = model(x)
padding = model.padding

x = torch.tensor([[[[[-1.3328, -0.4948],
                    [ 0.8689,  1.1423]],
                    [[-0.2671, -1.0868],
                    [ 1.3011,  1.0469]]]]])
model = nn.CircularPad3d((1, 1, 1, 1, 1, 1))
result = model(x)
padding = model.padding

# torch.nn.ConstantPad1d
x = torch.tensor([[[0., 1., 2., 3.], [4., 5., 6., 7.]]])
model = nn.ConstantPad1d(2, 3.5)
result = model(x)
padding = model.padding
value = model.value

x = torch.tensor([[[0., 1., 2., 3.], [4., 5., 6., 7.]]])
model = nn.ConstantPad1d((2, 1), 3.5)
result = model(x)
padding = model.padding
value = model.value

# torch.nn.ConstantPad2d
x = torch.tensor([[[[-0.4106,  0.1677], [-0.6648, -0.5669]]]])
model = nn.ConstantPad2d(1, 4.7)
result = model(x)
padding = model.padding
value = model.value

x = torch.tensor([[[[-0.4106,  0.1677], [-0.6648, -0.5669]]]])
model = nn.ConstantPad2d((1, 1, 1, 0), 4.8)
result = model(x)
padding = model.padding
value = model.value

# torch.nn.ConstantPad3d
x = torch.tensor([[[[[-1.3328, -0.4948],
                    [ 0.8689,  1.1423]],
                    [[-0.2671, -1.0868],
                    [ 1.3011,  1.0469]]]]])
model = nn.ConstantPad3d(1, 4.5)
result = model(x)
padding = model.padding
value = model.value

x = torch.tensor([[[[[-1.3328, -0.4948],
                    [ 0.8689,  1.1423]],
                    [[-0.2671, -1.0868],
                    [ 1.3011,  1.0469]]]]])
model = nn.ConstantPad3d((1, 1, 1, 1, 1, 1), 4.6)
result = model(x)
padding = model.padding
value = model.value

# torch.nn.Conv1d
x = torch.randn(20, 16, 50)
model = nn.Conv1d(16, 33, 3, stride=2, bias=False)
result = model(x)

x = torch.randn(20, 16, 50)
model = nn.Conv1d(16, 33, 3, stride=2, padding=4, bias=False)
result = model(x)

# torch.nn.Conv2d
x = torch.zeros(20, 16, 50, 100)
model = nn.Conv2d(16, 33, 3, stride=2, bias=False)
result = model(x)

x = torch.zeros(20, 16, 50, 100)
model = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), bias=False)
result = model(x)

# torch.nn.Conv3d
x = torch.rand(2, 16, 50, 20, 20)
model = nn.Conv3d(16, 33, 3, stride=2, bias=False)
result = model(x)

x = torch.randn(2, 16, 50, 20, 20)
model = nn.Conv3d(16, 33, (3, 3, 5), stride=(2, 2, 1), padding=(4, 2, 2), bias=False)
result = model(x)

# torch.nn.CosineSimilarity
x = torch.tensor([[1., 2., 3.], [2., 3., 4.]])
y = torch.tensor([[8., 3., 3.], [2., 3., 4.]])
model = nn.CosineSimilarity()
result = model(x, y)

x = torch.tensor([[1., 2., 3.], [2., 3., 4.]])
y = torch.tensor([[8., 3., 3.], [2., 3., 4.]])
model = nn.CosineSimilarity(0)
result = model(x, y)

# torch.nn.Dropout
x = torch.randn(20, 16)
model = nn.Dropout(0.4)
result = model(x)

x = torch.randn(20, 16)
model = nn.Dropout(0.4, False)
result = model(x)

# torch.nn.Embedding
embedding = torch.nn.Embedding(4, 3)
w0 = torch.Tensor([[0., 0., 0.],
            [1., 1., 1.],
            [2., 2., 2.],
            [3., 3., 3.]])
with torch.no_grad():
    embedding.weight[0]=w0[0]
    embedding.weight[1]=w0[1]
    embedding.weight[3]=w0[3]
x = torch.LongTensor([[0],[1],[3]])
result = embedding(x)

padding_idx = 0
embedding = torch.nn.Embedding(4, 3,padding_idx=padding_idx)
w0 = torch.Tensor([[0., 0., 0.],
            [1., 1., 1.],
            [2., 2., 2.],
            [3., 3., 3.]])
with torch.no_grad():
    embedding.weight[0]=w0[0]
    embedding.weight[1]=w0[1]
    embedding.weight[2]=w0[2]
    embedding.weight[3]=w0[3]
x = torch.LongTensor([[0],[1],[3]])
result = embedding(x)

# torch.nn.Fold
x = torch.ones([2, 3*2*2, 12])
fold = nn.Fold([4, 5], 2)
result = fold(x)

x = torch.ones([2, 3*2*2, 40])
fold = nn.Fold([4, 5], 2, 1, [1, 2], 1)
result = fold(x)

# torch.nn.GELU
x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
model = nn.GELU()
result = model(x)

x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
model = nn.GELU(approximate ='tanh')
result = model(x)

# torch.nn.GLU
x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
m = torch.nn.GLU()
result = m(x)

x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
m = torch.nn.GLU(dim=-1)
result = m(x)

# torch.nn.GroupNorm
a = torch.tensor([[[[2.,3.], [3., 5.]], [[5.,3.], [9., 5.]]]])
m = torch.nn.GroupNorm(2, 2)
result = m(a)

a = torch.tensor([[[[2.,3.], [3., 5.]], [[5.,3.], [9., 5.]]]])
m = torch.nn.GroupNorm(2, 2, eps=1e-05, affine=False)
result = m(a)

# torch.nn.Hardshrink
x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
model = nn.Hardshrink(0.8)
result = model(x)

x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
model = nn.Hardshrink(0.4)
result = model(x)

# torch.nn.HuberLoss
input = torch.tensor([[-1.2837, -0.0297,  0.0355],
    [ 0.9112, -1.7526, -0.4061]])
target = torch.tensor([[1., 2., 1.], [1., 2., 3.]])
loss = torch.nn.HuberLoss()
result = loss(input, target)

input = torch.tensor([[-1.2837, -0.0297,  0.0355],
    [ 0.9112, -1.7526, -0.4061]])
target = torch.tensor([[1., 2., 1.],[1., 2., 3.]])
loss = torch.nn.HuberLoss(reduction='mean')
result = loss(input, target)

# torch.nn.Identity
m = nn.Identity(20)
input = torch.ones(128, 20)
result = m(input)

m = nn.Identity(54, unused_argument1=0.1, unused_argument2=False)
input = torch.ones(128, 20)
output = m(input)
result = m(input)

# torch.nn.LPPool1d
input = torch.tensor([[[-0.5743,  0.4889, -0.0878,  0.4210, -0.0844],
    [ 0.3614,  0.8458, -0.6152,  0.6894,  0.2927],
    [-0.0087,  0.1098,  0.1783, -0.6953,  0.5519],
    [ 0.3789, -0.0560, -0.4090, -0.1070, -1.0139],
    [ 0.9204,  1.0817, -2.6126,  0.4244,  0.3272]]])
pool = nn.LPPool1d(1, 3, stride=2)
result = pool(input)

input = torch.tensor([[[ 0.6430,  0.4511, -1.6757,  1.7116],
    [-0.2288, -0.4111, -1.3602,  0.2685],
    [ 0.2363,  1.9341,  0.8522, -0.1846],
    [ 1.6496, -0.0675, -0.7208, -1.0018]],

    [[-0.3183,  0.8029, -0.4993,  1.0598],
    [-0.4952, -0.9536,  0.1954,  0.0551],
    [ 1.2257,  0.7517,  0.4063, -1.2151],
    [-1.3562,  0.3547,  1.1147,  1.2898]],

    [[ 0.1205, -0.1889,  0.5086, -0.8080],
    [ 0.3156, -0.8298,  2.0242, -0.9184],
    [-0.4005,  1.3586,  0.6205, -0.7487],
    [ 1.6239,  0.2900,  0.9671,  1.2961]],

    [[-1.1996, -0.2201, -0.9466, -0.7264],
    [-0.0313,  0.8284, -0.3588,  1.3522],
    [-0.0991, -0.5112, -0.1785,  2.0903],
    [-1.3286, -0.9333, -0.1404,  1.2582]]])
pool = nn.LPPool1d(2, 4, stride=2)
result = pool(input)

# torch.nn.LPPool2d
input = torch.tensor([[[[-0.5743,  0.4889, -0.0878,  0.4210, -0.0844],
    [ 0.3614,  0.8458, -0.6152,  0.6894,  0.2927],
    [-0.0087,  0.1098,  0.1783, -0.6953,  0.5519],
    [ 0.3789, -0.0560, -0.4090, -0.1070, -1.0139],
    [ 0.9204,  1.0817, -2.6126,  0.4244,  0.3272]]]])
pool = nn.LPPool2d(1, 3, stride=2)
result = pool(input)

input = torch.tensor([[[[ 0.6430,  0.4511, -1.6757,  1.7116],
    [-0.2288, -0.4111, -1.3602,  0.2685],
    [ 0.2363,  1.9341,  0.8522, -0.1846],
    [ 1.6496, -0.0675, -0.7208, -1.0018]],

    [[-0.3183,  0.8029, -0.4993,  1.0598],
    [-0.4952, -0.9536,  0.1954,  0.0551],
    [ 1.2257,  0.7517,  0.4063, -1.2151],
    [-1.3562,  0.3547,  1.1147,  1.2898]],

    [[ 0.1205, -0.1889,  0.5086, -0.8080],
    [ 0.3156, -0.8298,  2.0242, -0.9184],
    [-0.4005,  1.3586,  0.6205, -0.7487],
    [ 1.6239,  0.2900,  0.9671,  1.2961]],

    [[-1.1996, -0.2201, -0.9466, -0.7264],
    [-0.0313,  0.8284, -0.3588,  1.3522],
    [-0.0991, -0.5112, -0.1785,  2.0903],
    [-1.3286, -0.9333, -0.1404,  1.2582]]]])
pool = nn.LPPool2d(2, 4, stride=2)
result = pool(input)

# torch.nn.LayerNorm
m = nn.LayerNorm(10)
input = torch.ones(2, 5, 10)
result = m(input)

m = nn.LayerNorm([5,10,10])
input = torch.ones(2, 5, 10, 10)
result = m(input)

# torch.nn.LeakyReLU
x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
model = nn.LeakyReLU()
result = model(x)

x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
model = nn.LeakyReLU(0.4)
result = model(x)

# torch.nn.LocalResponseNorm
lrn = nn.LocalResponseNorm(2)
signal_2d = torch.randn(32, 5, 24, 24)
result = lrn(signal_2d)

lrn = nn.LocalResponseNorm(2)
signal_4d = torch.randn(16, 5, 7, 7, 7)
result = lrn(signal_4d)

# torch.nn.LogSigmoid
x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
model = nn.LogSigmoid()
result = model(x)

x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
result = nn.LogSigmoid()(x)

# torch.nn.MaxPool1d
x = torch.tensor([[[0., 1., 2., 3.], [4., 5., 6., 7.]]])
model = nn.MaxPool1d(2)
result = model(x)

x = torch.tensor([[[0., 1., 2., 3.], [4., 5., 6., 7.]]])
model = nn.MaxPool1d(2, 1)
result = model(x)

# torch.nn.MaxPool2d
x = torch.tensor([[[[0., 1., 2., 3.], [4., 5., 6., 7.]]]])
model = nn.MaxPool2d(2)
result = model(x)

x = torch.tensor([[[[0., 1., 2., 3.], [4., 5., 6., 7.]]]])
model = nn.MaxPool2d(2, 1)
result = model(x)

# torch.nn.MaxPool3d
x = torch.tensor([[[[[-0.8658,  1.0869, -2.1977],
   [-2.1073,  1.0974, -1.4485],
   [ 0.5880, -0.7189,  0.1089]],

  [[ 1.3036,  0.3086, -1.2245],
   [-0.6707, -0.0195, -0.1474],
   [ 0.2727, -0.4938, -0.6854]],

  [[ 0.5525,  1.0111, -0.1847],
   [ 0.1111, -0.6373, -0.2220],
   [-0.5963,  0.7734,  0.0409]]]]])
model = nn.MaxPool3d(2)
result = model(x)

x = torch.tensor([[[[[-0.8658,  1.0869, -2.1977],
   [-2.1073,  1.0974, -1.4485],
   [ 0.5880, -0.7189,  0.1089]],

  [[ 1.3036,  0.3086, -1.2245],
   [-0.6707, -0.0195, -0.1474],
   [ 0.2727, -0.4938, -0.6854]],

  [[ 0.5525,  1.0111, -0.1847],
   [ 0.1111, -0.6373, -0.2220],
   [-0.5963,  0.7734,  0.0409]]]]])
model = nn.MaxPool3d((2,1,1), 1)
result = model(x)

# torch.nn.MaxUnpool1d
pool = nn.MaxPool1d(2, stride=2, return_indices=True)
unpool = nn.MaxUnpool1d(2, 2)
input = torch.tensor([[[1., 2, 3, 4, 5, 6, 7, 8]]])
output, indices = pool(input)
result = unpool(output, indices)

pool = nn.MaxPool1d(2, stride=1, return_indices=True)
unpool = nn.MaxUnpool1d(2, stride=1)
input = torch.tensor([[[1., 2, 3, 4, 5, 6, 7, 8]]])
output, indices = pool(input)
result = unpool(output, indices)

# torch.nn.MaxUnpool2d
pool = nn.MaxPool2d(2, stride=2, return_indices=True)
unpool = nn.MaxUnpool2d(2, stride=2)
input = torch.tensor([[[[ 1.,  2.,  3.,  4.],
                    [ 5.,  6.,  7.,  8.],
                    [ 9., 10., 11., 12.],
                    [13., 14., 15., 16.]]]])
output, indices = pool(input)
result =unpool(output, indices)

pool = nn.MaxPool2d(2, stride=2, return_indices=True)
unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
input = torch.tensor([[[[ 1.,  2.,  3.,  4.],
                    [ 5.,  6.,  7.,  8.],
                    [ 9., 10., 11., 12.],
                    [13., 14., 15., 16.]]]])
output, indices = pool(input)
result =unpool(output, indices)

# torch.nn.MaxUnpool3d
pool = nn.MaxPool3d(3, stride=2, return_indices=True)
unpool = nn.MaxUnpool3d(3, stride=2)
output, indices = pool(torch.ones(2, 16, 51, 33, 15))
result = unpool(output, indices)

pool = nn.MaxPool3d(3, stride=2, return_indices=True)
unpool = nn.MaxUnpool3d(3, 2)
output, indices = pool(torch.ones(2, 16, 51, 33, 15))
result = unpool(output, indices)

# torch.nn.Module
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

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

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = MLP(784, 256, 10)
result = model.__class__.__name__

# torch.nn.Module.add_module
x = torch.tensor([1., 2., 3.])
module1 = torch.nn.Module()
module1.register_buffer('buffer', x)
module2 = torch.nn.Module()
module2.add_module('submodule', module1)
result = module2.submodule.buffer

x = torch.tensor([1., 2., 3.])
module1 = torch.nn.Module()
module1.register_buffer('buffer', x)
module2 = torch.nn.Module()
module2.add_module(name='submodule', module=module1)
result = module2.submodule.buffer

# torch.nn.Module.apply
def init_weights(m):
    pass
net = nn.Sequential(nn.Linear(2, 2,bias=False), nn.Linear(2, 2, bias=False))
net.apply(init_weights)
a =torch.tensor([0.,0.])
result = net(a)

def init_weights(m):
    pass
net = nn.Sequential(nn.Linear(2, 2,bias=False), nn.Linear(2, 2, bias=False))
net.apply(fn=init_weights)
a =torch.tensor([0.,0.])
result = net(a)

# torch.nn.Module.bfloat16
x = torch.tensor([1., 2., 3.])
module1 = torch.nn.Module()
module1.register_buffer('buffer', x)
module1.bfloat16()
result = module1.buffer

# torch.nn.Module.buffers
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.register_buffer('buf1', torch.tensor([1.,2.,4.,5.]))
        self.register_buffer('buf2', torch.tensor([1.,2.,4.,5.]))
        self.register_buffer('buf3', torch.tensor([1.,2.,4.,5.]))

    def forward(self, x):
        pass

model = Model()
result = []
for buf in model.buffers():
    result.append(buf)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.register_buffer('buf1', torch.tensor([1.,2.,4.,5.]))
        self.register_buffer('buf2', torch.tensor([1.,2.,4.,5.]))
        self.register_buffer('buf3', torch.tensor([1.,2.,4.,5.]))

    def forward(self, x):
        pass

model = Model()
result = []
for buf in model.buffers(True):
    result.append(buf)

# torch.nn.Module.children
l = nn.Linear(2, 2, bias=False)
net = nn.Sequential(l, l)
result = torch.Tensor([0,0])
for i,j in enumerate(net.children()):
    result = j(result)

# torch.nn.Module.cpu
x = torch.tensor([1., 2., 3.])
module1 = torch.nn.Module()
module1.register_buffer('buffer', x)
module1.cpu()
result = module1.buffer

# torch.nn.Module.cuda
x = torch.tensor([1., 2., 3.])
module1 = torch.nn.Module()
module1.register_buffer('buffer', x)
result = module1.cuda()

x = torch.tensor([1., 2., 3.])
module1 = torch.nn.Module()
module1.register_buffer('buffer', x)
result = module1.cuda(device=0)

# torch.nn.Module.double
x = torch.tensor([1., 2., 3.])
module1 = torch.nn.Module()
module1.register_buffer('buffer', x)
module1.double()
result = module1.buffer

# torch.nn.Module.eval
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

# torch.nn.Module.float
x = torch.tensor([1., 2., 3.])
module1 = torch.nn.Module()
module1.register_buffer('buffer', x)
module1.float()
result = module1.buffer

# torch.nn.Module.get_buffer
x = torch.tensor([1., 2., 3.])
module1 = torch.nn.Module()
module1.register_buffer('buffer', x)
result = module1.get_buffer('buffer')

x = torch.tensor([1., 2., 3.])
module1 = torch.nn.Module()
module1.register_buffer('buffer', x)
result = module1.get_buffer(target='buffer')

# torch.nn.Module.get_parameter
x = torch.tensor([1., 2., 3.])
module1 = torch.nn.Module()
module1.register_parameter('param1', torch.nn.parameter.Parameter(x))
result = module1.get_parameter('param1')

x = torch.tensor([1., 2., 3.])
module1 = torch.nn.Module()
module1.register_parameter('param1', torch.nn.parameter.Parameter(x))
result = module1.get_parameter(target='param1')

# torch.nn.Module.get_submodule
x = torch.tensor([1., 2., 3.])
module1 = torch.nn.Module()
module1.register_buffer('buffer', x)
module2 = torch.nn.Module()
module2.register_module('submodule', module1)
result = module2.get_submodule('submodule')

x = torch.tensor([1., 2., 3.])
module1 = torch.nn.Module()
module1.register_buffer('buffer', x)
module2 = torch.nn.Module()
module2.register_module(name='submodule', module=module1)
result = module2.get_submodule(target='submodule')

# torch.nn.Module.half
x = torch.tensor([1., 2., 3.])
module1 = torch.nn.Module()
module1.register_buffer('buffer', x)
module1.half()
result = module1.buffer

# torch.nn.Module.load_state_dict
class TheModelClass(torch.nn.Module):
    def forward(self, x):
        return x

a = torch.tensor([[[[1.,2.,3.,4.]]]])
model = TheModelClass()
PATH = './tensor.pt'
torch.save(model.state_dict(), PATH)
model.load_state_dict(torch.load(PATH))
result = model(a)

class TheModelClass(torch.nn.Module):
    def forward(self, x):
        return x

a = torch.tensor([[[[1.,2.,3.,4.]]]])
model = TheModelClass()
PATH = './tensor.pt'
torch.save(model.state_dict(), PATH)
model.load_state_dict(torch.load(PATH), True)
result = model(a)

# torch.nn.Module.modules
l = nn.Linear(2, 2, bias=False)
net = nn.Sequential(l, l)
result = torch.Tensor([0,0])
for i,j in enumerate(net.modules()):
    result = j(result)

# torch.nn.Module.named_children
l = nn.Linear(2, 2,bias=False)
l1 = nn.Linear(2, 2,bias=False)
model = nn.Sequential(OrderedDict([
                ('wfs', l),
                ('wfs1', l1)
                ]))
result = torch.Tensor([0,0])
for name, module in model.named_children():
    result = module(result)

# torch.nn.Module.named_modules
l = nn.Linear(2, 2)
net = nn.Sequential(OrderedDict([
                ('wfs', l),
                ('wfs1', l),
                ('wfs', l),
                ('wfs1', l)]
                ))
z = net.named_modules(prefix="wfs", remove_duplicate=True)
name_list = []
for idx,m in enumerate(z):
    name_list.append(m[0])
result = name_list

l = nn.Linear(2, 2)
net = nn.Sequential(OrderedDict([
                ('wfs', l),
                ('wfs1', l)
                ]))
z = net.named_modules(prefix="wfs", remove_duplicate=False)
name_list = []
for idx,m in enumerate(z):
    name_list.append(m[0])
result = name_list

# torch.nn.Module.named_parameters
result = []
class TestForHook(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear_1 = nn.Linear(in_features=2, out_features=2)
    def forward(self, x):
        x1 = self.linear_1(x)
        return x, x, x1
a = TestForHook()
for a,b in a.named_parameters(prefix="wfs"):
    result.append(b)
result = result[0]

result = []
class TestForHook(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear_1 = nn.Linear(in_features=2, out_features=2)
    def forward(self, x):
        x1 = self.linear_1(x)
        return x, x, x1
a = TestForHook()
for a,b in a.named_parameters(prefix="wfs", recurse=True):
    result.append(b)
result = result[0]

# torch.nn.Module.parameters
model= nn.ReLU()
list = model.parameters()
result = []
for i in list:
    result.append(i)

model= nn.Conv2d(1, 20, 5)
list = model.parameters()
result = []
for i in list:
    result.append(i)
weight, bias = result[0], result[1]

# torch.nn.Module.register_buffer
x = torch.tensor([1., 2., 3.])
module = torch.nn.Module()
module.register_buffer('buffer', x)
result = module.buffer

x = torch.tensor([1., 2., 3.])
module = torch.nn.Module()
module.register_buffer(name='buffer', tensor=x, persistent=True)
result = module.buffer

# torch.nn.Module.register_module
x = torch.tensor([1., 2., 3.])
module1 = torch.nn.Module()
module1.register_buffer('buffer', x)
module2 = torch.nn.Module()
module2.register_module('submodule', module1)
result = module2.submodule.buffer

x = torch.tensor([1., 2., 3.])
module1 = torch.nn.Module()
module1.register_buffer('buffer', x)
module2 = torch.nn.Module()
module2.register_module(name='submodule', module=module1)
result = module2.submodule.buffer

# torch.nn.Module.register_parameter
x = torch.tensor([1., 2., 3.])
module1 = torch.nn.Module()
module1.register_parameter('param1', torch.nn.parameter.Parameter(x))
result = module1.param1

x = torch.tensor([1., 2., 3.])
module1 = torch.nn.Module()
module1.register_parameter(name='param1', param=torch.nn.parameter.Parameter(x))
result = module1.param1

# torch.nn.Module.requires_grad_
x = torch.tensor([1., 2., 3.])
module1 = torch.nn.Module()
module1.register_buffer('buffer', x)
module1.requires_grad_(True)
result = module1.buffer

x = torch.tensor([1., 2., 3.])
module1 = torch.nn.Module()
module1.register_buffer('buffer', x)
module1.requires_grad_(requires_grad=True)
result = module1.buffer

# torch.nn.Module.state_dict
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 2)

    def forward(self, x):
        self.x = torch.relu(self.fc1(x))
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
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = Net()

state_dict = model.state_dict(prefix="wfs")

result = []
for key, value in state_dict.items():
    result.append(key)

# torch.nn.Module.to
x = torch.tensor([1., 2., 3.])
module1 = torch.nn.Module()
module1.register_buffer('buffer', x)
module1.to(dtype=torch.float32)
result = module1.buffer

x = torch.tensor([1., 2., 3.])
module1 = torch.nn.Module()
module1.register_buffer('buffer', x)
module1.to(device="cpu")
result = module1.buffer

# torch.nn.Module.train
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

# torch.nn.Module.xpu
x = torch.tensor([1., 2., 3.])
module1 = torch.nn.Module()
module1.register_buffer('buffer', x)
module1.xpu()
result = None

# torch.nn.Module.zero_grad
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

# torch.nn.ModuleDict
choices = nn.ModuleDict({
        'relu': nn.ReLU()
})
i = torch.tensor([1.,2.])
result = choices['relu'](i)

choices = nn.ModuleDict(modules={
        'relu': nn.ReLU()
})
i = torch.tensor([1.,2.])
result = choices['relu'](i)

# torch.nn.ModuleList
choices = nn.ModuleList([
        nn.ReLU()
])
i = torch.tensor([1.,2.])
result = choices[0](i)

choices = nn.ModuleList(modules=[
        nn.ReLU()
])
i = torch.tensor([1.,2.])
result = choices[0](i)

# torch.nn.PairwiseDistance
x = torch.tensor([[1., 2., 3.], [2., 3., 4.]])
y = torch.tensor([[8., 3., 3.], [1.4, 3.6, 0.8]])
model = nn.PairwiseDistance()
result = model(x, y)

x = torch.tensor([[1., 2., 3.], [2., 3., 4.]])
y = torch.tensor([[8., 3., 3.], [1.4, 3.6, 0.8]])
model = nn.PairwiseDistance(3)
result = model(x, y)

# torch.nn.Parameter
x = torch.tensor([[1., 2., 3.], [2., 3., 4.]])
result = torch.nn.Parameter(x)

x = torch.tensor([[1., 2., 3.], [2., 3., 4.]])
result = torch.nn.Parameter(x, requires_grad=False)

# torch.nn.PixelShuffle
x = torch.ones(1, 9, 4, 4)
model = nn.PixelShuffle(3)
result = model(x)

x = torch.ones(1, 9, 4, 4)
model = nn.PixelShuffle(upscale_factor=3)
result = model(x)

# torch.nn.PixelUnshuffle
x = torch.ones(1, 9, 12, 12)
model = nn.PixelUnshuffle(3)
result = model(x)

x = torch.ones(1, 9, 12, 12)
model = nn.PixelUnshuffle(downscale_factor=3)
result = model(x)

# torch.nn.ReLU
x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
model = nn.ReLU()
result = model(x)

x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
model = nn.ReLU(False)
result = model(x)

# torch.nn.ReflectionPad1d
x = torch.tensor([[[0., 1., 2., 3.], [4., 5., 6., 7.]]])
model = nn.ReflectionPad1d(2)
result = model(x)
padding = model.padding

x = torch.tensor([[[0., 1., 2., 3.], [4., 5., 6., 7.]]])
model = nn.ReflectionPad1d((2, 1))
result = model(x)
padding = model.padding

# torch.nn.ReflectionPad2d
x = torch.tensor([[[[-0.4106,  0.1677], [-0.6648, -0.5669]]]])
model = nn.ReflectionPad2d(1)
result = model(x)
padding = model.padding

x = torch.tensor([[[[-0.4106,  0.1677], [-0.6648, -0.5669]]]])
model = nn.ReflectionPad2d((1, 1, 1, 0))
result = model(x)
padding = model.padding

# torch.nn.ReflectionPad3d
x = torch.tensor([[[[[-1.3328, -0.4948],
                    [ 0.8689,  1.1423]],
                    [[-0.2671, -1.0868],
                    [ 1.3011,  1.0469]]]]])
model = nn.ReflectionPad3d(1)
result = model(x)
padding = model.padding

x = torch.tensor([[[[[-1.3328, -0.4948],
                    [ 0.8689,  1.1423]],
                    [[-0.2671, -1.0868],
                    [ 1.3011,  1.0469]]]]])
model = nn.ReflectionPad3d((1, 1, 1, 1, 1, 1))
result = model(x)
padding = model.padding

# torch.nn.ReplicationPad1d
x = torch.tensor([[[0., 1., 2., 3.], [4., 5., 6., 7.]]])
model = nn.ReplicationPad1d(2)
result = model(x)

x = torch.tensor([[[0., 1., 2., 3.], [4., 5., 6., 7.]]])
model = nn.ReplicationPad1d((2, 1))
result = model(x)

# torch.nn.ReplicationPad2d
x = torch.tensor([[[[-0.4106,  0.1677], [-0.6648, -0.5669]]]])
model = nn.ReplicationPad2d(1)
result = model(x)

x = torch.tensor([[[[-0.4106,  0.1677], [-0.6648, -0.5669]]]])
model = nn.ReplicationPad2d((1, 1, 1, 0))
result = model(x)

# torch.nn.ReplicationPad3d
x = torch.tensor([[[[[-1.3328, -0.4948],
                    [ 0.8689,  1.1423]],
                    [[-0.2671, -1.0868],
                    [ 1.3011,  1.0469]]]]])
model = nn.ReplicationPad3d(1)
result = model(x)

x = torch.tensor([[[[[-1.3328, -0.4948],
                    [ 0.8689,  1.1423]],
                    [[-0.2671, -1.0868],
                    [ 1.3011,  1.0469]]]]])
model = nn.ReplicationPad3d((1, 1, 1, 1, 1, 1))
result = model(x)

# torch.nn.Sequential
m = torch.nn.Sequential(torch.nn.ReLU())
result = m(torch.tensor([-1., 2., 3., 4.]))

m = torch.nn.Sequential(*[torch.nn.ReLU()])
result = m(torch.tensor([-1., 2., 3., 4.]))

# torch.nn.SiLU
x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
model = nn.SiLU()
result = model(x)

x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
model = nn.SiLU(False)
result = model(x)

# torch.nn.Sigmoid
x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
model = nn.Sigmoid()
result = model(x)

x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
result = nn.Sigmoid()(x)

# torch.nn.Softplus
x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
model = nn.Softplus()
result = model(x)

x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
model = nn.Softplus(2, 20)
result = model(x)

# torch.nn.Softshrink
x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
model = nn.Softshrink()
result = model(x)

x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
model = nn.Softshrink(0.7)
result = model(x)

# torch.nn.Softsign
x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
model = nn.Softsign()
result = model(x)

x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
result = nn.Softsign()(x)

# torch.nn.Tanh
x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
model = nn.Tanh()
result = model(x)

x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
result = nn.Tanh()(x)

# torch.nn.Tanhshrink
x = torch.tensor([-0.4, -0.2, 0.1, 0.3])
model = nn.Tanhshrink()
result = model(x)

x = torch.tensor([-0.4, -0.2, 0.1, 0.3])
result = nn.Tanhshrink()(x)

# torch.nn.TransformerDecoder
decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
memory = torch.rand(20, 32, 512)
tgt = torch.rand(20, 32, 512)
result = transformer_decoder(tgt, memory)

decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
transformer_decoder = nn.TransformerDecoder(decoder_layer, 6)
memory = torch.rand(20, 32, 512)
tgt = torch.rand(20, 32, 512)
result = transformer_decoder(tgt, memory)

# torch.nn.TripletMarginWithDistanceLoss
x = torch.tensor([[1., 5, 3], [0, 3, 2], [1, 4, 1]])
positive = torch.tensor([[5., 1, 2], [3, 2, 1], [3, -1, 1]])
negative = torch.tensor([[2., 1, -3], [1, 1, -1], [4, -2, 1]])
model = nn.TripletMarginWithDistanceLoss()
result = model(x, positive, negative)

x = torch.tensor([[1., 5, 3], [0, 3, 2], [1, 4, 1]])
positive = torch.tensor([[5., 1, 2], [3, 2, 1], [3, -1, 1]])
negative = torch.tensor([[2., 1, -3], [1, 1, -1], [4, -2, 1]])
model = nn.TripletMarginWithDistanceLoss(margin=2)
result = model(x, positive, negative)

# torch.nn.Upsample
input = torch.tensor([[[[ 1.1524,  0.4714,  0.2857],
 [-1.2533, -0.9829, -1.0981],
 [ 0.1507, -1.1431, -2.0361]],

[[ 0.1024, -0.4482,  0.4137],
 [ 0.9385,  0.4565,  0.7702],
 [ 0.4135, -0.2587,  0.0482]]]])
m = torch.nn.Upsample(scale_factor=2, mode='nearest')
result = m(input)

input = torch.tensor([[[[ 1.1524,  0.4714,  0.2857],
 [-1.2533, -0.9829, -1.0981],
 [ 0.1507, -1.1431, -2.0361]],

[[ 0.1024, -0.4482,  0.4137],
 [ 0.9385,  0.4565,  0.7702],
 [ 0.4135, -0.2587,  0.0482]]]])
m = torch.nn.Upsample(scale_factor=2, mode='bilinear')
result = m(input)

# torch.nn.UpsamplingBilinear2d
x = torch.tensor([[[[1., 2.], [3., 4.]]]])
model = nn.UpsamplingBilinear2d(scale_factor=2)
result = model(x)

x = torch.tensor([[[[1., 2.], [3., 4.]]]])
model = nn.UpsamplingBilinear2d(size=4)
result = model(x)

# torch.nn.UpsamplingNearest2d
x = torch.tensor([[[[1., 2.], [3., 4.]]]])
model = nn.UpsamplingNearest2d(scale_factor=2)
result = model(x)

x = torch.tensor([[[[1., 2.], [3., 4.]]]])
model = nn.UpsamplingNearest2d(size=4)
result = model(x)

# torch.nn.ZeroPad1d
x = torch.tensor([[[-0.4106,  0.1677], [-0.6648, -0.5669]]])
model = nn.ZeroPad1d(1)
result = model(x)

x = torch.tensor([[[-0.4106,  0.1677], [-0.6648, -0.5669]]])
model = nn.ZeroPad1d((1, 1, 1, 0))
result = model(x)

# torch.nn.ZeroPad2d
x = torch.tensor([[[[-0.4106,  0.1677], [-0.6648, -0.5669]]]])
model = nn.ZeroPad2d(1)
result = model(x)

x = torch.tensor([[[[-0.4106,  0.1677], [-0.6648, -0.5669]]]])
model = nn.ZeroPad2d((1, 1, 1, 0))
result = model(x)

# torch.nn.ZeroPad3d
x = torch.tensor([[[[[-0.4106,  0.1677], [-0.6648, -0.5669]]]]])
model = nn.ZeroPad3d(1)
result = model(x)

x = torch.tensor([[[[[-0.4106,  0.1677], [-0.6648, -0.5669]]]]])
model = nn.ZeroPad3d((1, 1, 1, 0))
result = model(x)

# torch.nn.attention.sdpa_kernel
modified_backend_state = [torch.nn.attention.SDPBackend.MATH]

np.random.seed(100)
x_data = np.random.randn(2, 2, 2, 2)
x = torch.tensor(x_data, dtype=torch.float32)

with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
    result = torch.nn.functional.scaled_dot_product_attention(x, x, x).to(torch.float32)
    current_backends = torch.nn.attention._cur_sdpa_kernel_backends()
    assert current_backends == modified_backend_state

original_backend_state = set(torch.nn.attention._cur_sdpa_kernel_backends())
modified_backend_state = [torch.nn.attention.SDPBackend.MATH]

np.random.seed(100)
x_data = np.random.randn(2, 2, 2, 2)
x = torch.tensor(x_data, dtype=torch.float32)

# Check original state
current_backends = set(torch.nn.attention._cur_sdpa_kernel_backends())
assert current_backends == original_backend_state, f"Expected {original_backend_state}, got {current_backends}"

with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
    output1 = torch.nn.functional.scaled_dot_product_attention(x, x, x).to(torch.float32)
    current_backends = torch.nn.attention._cur_sdpa_kernel_backends()
    assert current_backends == modified_backend_state, f"Expected {modified_backend_state}, got {current_backends}"

    output2 = torch.nn.functional.scaled_dot_product_attention(x, x, x).to(torch.float32)
    current_backends = torch.nn.attention._cur_sdpa_kernel_backends()
    assert current_backends == modified_backend_state, f"Expected {modified_backend_state}, got {current_backends}"

    output3 = torch.nn.functional.scaled_dot_product_attention(x, x, x).to(torch.float32)
    current_backends = torch.nn.attention._cur_sdpa_kernel_backends()
    assert current_backends == modified_backend_state, f"Expected {modified_backend_state}, got {current_backends}"

# Check back to original state
current_backends = set(torch.nn.attention._cur_sdpa_kernel_backends())
assert current_backends == original_backend_state, f"Expected {original_backend_state}, got {current_backends}"

result = output1 + output2 + output3

# torch.nn.functional.adaptive_avg_pool1d
x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
result = F.adaptive_avg_pool1d(x, 5)

x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
result = F.adaptive_avg_pool1d(input=x, output_size=5)

# torch.nn.functional.adaptive_avg_pool2d
x = torch.tensor([[[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]]])
result = F.adaptive_avg_pool2d(x, 4)

x = torch.tensor([[[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]]])
result = F.adaptive_avg_pool2d(x, output_size=4)

# torch.nn.functional.adaptive_avg_pool3d
x = torch.tensor([[[[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]]]])
result = F.adaptive_avg_pool3d(x, 4)

x = torch.tensor([[[[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]]]])
result = F.adaptive_avg_pool3d(x, output_size=4)

# torch.nn.functional.adaptive_max_pool1d
x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
result = nn.functional.adaptive_max_pool1d(x, 5)

x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
result = nn.functional.adaptive_max_pool1d(x, output_size=5)

# torch.nn.functional.adaptive_max_pool2d
x = torch.tensor([[[[ 0.9785,  1.2013,  2.4873, -1.1891],
                    [-0.0832, -0.5456, -0.5009,  1.5103],
                    [-1.2860,  1.0287, -1.3902,  0.4627],
                    [-0.0502, -1.3924, -0.3327,  0.1678]]]])
result = nn.functional.adaptive_max_pool2d(x, 5)

x = torch.tensor([[[[ 0.9785,  1.2013,  2.4873, -1.1891],
                    [-0.0832, -0.5456, -0.5009,  1.5103],
                    [-1.2860,  1.0287, -1.3902,  0.4627],
                    [-0.0502, -1.3924, -0.3327,  0.1678]]]])
result = nn.functional.adaptive_max_pool2d(x, output_size=2)

# torch.nn.functional.adaptive_max_pool3d
x = torch.tensor([[[[[-1.1494, -1.3829],
                    [ 0.4995, -1.3094]],
                    [[ 1.0015,  1.4919],
                    [-1.5187,  0.0235]]]]])
result = nn.functional.adaptive_max_pool3d(x, (1, 1, 1))

x = torch.tensor([[[[[-1.1494, -1.3829],
                    [ 0.4995, -1.3094]],
                    [[ 1.0015,  1.4919],
                    [-1.5187,  0.0235]]]]])
result = nn.functional.adaptive_max_pool3d(x, output_size=1)

# torch.nn.functional.conv1d
x = torch.randn(33, 16, 30)
weight = torch.randn(20, 16, 5)
result = F.conv1d(x, weight)

x = torch.randn(33, 16, 30)
weight = torch.randn(20, 16, 5)
bias = torch.randn(20)
result = F.conv1d(x, weight, bias)

# torch.nn.functional.conv2d
x = torch.randn(33, 16, 30, 30)
weight = torch.randn(20, 16, 5, 5)
result = F.conv2d(x, weight)

x = torch.randn(33, 16, 30, 30)
weight = torch.randn(20, 16, 5, 5)
bias = torch.randn(20)
result = F.conv2d(x, weight, bias)

# torch.nn.functional.conv3d
x = torch.randn(33, 16, 30, 30, 30)
weight = torch.randn(20, 16, 5, 5, 5)
result = F.conv3d(x, weight)

x = torch.randn(33, 16, 10, 10, 10)
weight = torch.randn(20, 16, 2, 2, 2)
bias = torch.randn(20)
result = F.conv3d(x, weight, bias)

# torch.nn.functional.cosine_similarity
x = torch.tensor([[1., 2., 3.], [2., 3., 4.]])
y = torch.tensor([[8., 3., 3.], [2., 3., 4.]])
result = F.cosine_similarity(x, y)

x = torch.tensor([[1., 2., 3.], [2., 3., 4.]])
y = torch.tensor([[8., 3., 3.], [2., 3., 4.]])
result = F.cosine_similarity(x, y, 1)

# torch.nn.functional.dropout
x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
result = F.dropout(x)

x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
result = F.dropout(x, 0.5)

# torch.nn.functional.dropout1d
x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
result = F.dropout1d(x)

x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
result = F.dropout1d(x, 0.5)

# torch.nn.functional.embedding
embedding_matrix = torch.Tensor([[0., 0., 0.],
            [1., 1., 1.],
            [2., 2., 2.],
            [3., 3., 3.]])

x = torch.tensor(np.array([[0,1],[2,3]]))
result = torch.nn.functional.embedding(x,embedding_matrix)

embedding_matrix = torch.Tensor([[0., 0., 0.],
            [1., 1., 1.],
            [2., 2., 2.],
            [3., 3., 3.]])

x = torch.tensor(np.array([[0,1],[2,3]]))
result = torch.nn.functional.embedding(x,embedding_matrix,padding_idx=0)

# torch.nn.functional.fold
x = torch.randn(1, 3 * 2 * 2, 12)
result = F.fold(x, output_size=(4, 5), kernel_size=(2, 2))

x = torch.randn(1, 3 * 2 * 2, 12)
result = F.fold(x, output_size=(4, 5), kernel_size=(2, 2), stride=1)

# torch.nn.functional.gelu
x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
result = F.gelu(x)

x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
result = F.gelu(x, approximate='tanh')

# torch.nn.functional.glu
x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
result = F.glu(x)

x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
result = F.glu(x, -1)

# torch.nn.functional.grid_sample
x = torch.tensor([[[[-0.6,  0.8, -0.5], [-0.5,  0.2,  1.2], [ 1.4,  0.3, -0.2]]]])
grid = torch.tensor([[[[ 0.2,  0.3],[-0.4, -0.3],[-0.9,  0.3],[-0.9, -0.6]],
                    [[ 0.4,  0.1],[ 0.9, -0.8],[ 0.4,  0.5],[ 0.5, -0.2]],
                    [[ 0.1, -0.8],[-0.3, -1. ],[ 0.7,  0.4],[ 0.2,  0.8]]]])
result = F.grid_sample(x, grid)

x = torch.tensor([[[[-0.6,  0.8, -0.5], [-0.5,  0.2,  1.2], [ 1.4,  0.3, -0.2]]]])
grid = torch.tensor([[[[ 0.2,  0.3],[-0.4, -0.3],[-0.9,  0.3],[-0.9, -0.6]],
                    [[ 0.4,  0.1],[ 0.9, -0.8],[ 0.4,  0.5],[ 0.5, -0.2]],
                    [[ 0.1, -0.8],[-0.3, -1. ],[ 0.7,  0.4],[ 0.2,  0.8]]]])
result = F.grid_sample(x, grid, padding_mode="border")

# torch.nn.functional.group_norm
x = torch.tensor([[[[-1.2392, -0.1310, -0.6679,  0.5476],
                    [ 1.1738, -1.7384, -0.7733,  0.3261],
                    [-0.0926, -1.0448, -1.2557, -1.5503],
                    [ 0.6402,  0.9072,  0.6780, -1.9885]],

                    [[ 0.0639, -1.1592,  1.4242, -0.4641],
                    [-0.1920,  0.1826,  1.9217, -0.4359],
                    [ 1.1926, -0.0247,  0.4744, -1.0216],
                    [-0.0360, -1.1656,  0.3661, -1.8147]]]])
result = F.group_norm(x, 2)

x = torch.tensor([[[[-1.2392, -0.1310, -0.6679,  0.5476],
                    [ 1.1738, -1.7384, -0.7733,  0.3261],
                    [-0.0926, -1.0448, -1.2557, -1.5503],
                    [ 0.6402,  0.9072,  0.6780, -1.9885]],

                    [[ 0.0639, -1.1592,  1.4242, -0.4641],
                    [-0.1920,  0.1826,  1.9217, -0.4359],
                    [ 1.1926, -0.0247,  0.4744, -1.0216],
                    [-0.0360, -1.1656,  0.3661, -1.8147]]]])
result = F.group_norm(x, 2)

# torch.nn.functional.hardshrink
x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
result = F.hardshrink(x)

x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
result = F.hardshrink(x, 0.8)

# torch.nn.functional.interpolate
x = torch.tensor([[[[1., 2., 3.], [2., 3., 4.]]]])
result = F.interpolate(x, scale_factor=2)

x = torch.tensor([[[[1., 2., 3.], [2., 3., 4.]]]])
result = F.interpolate(x, size=(2,3))

# torch.nn.functional.layer_norm
input = torch.tensor([[[ 1.1524,  0.4714,  0.2857],
 [-1.2533, -0.9829, -1.0981],
 [ 0.1507, -1.1431, -2.0361]],

[[ 0.1024, -0.4482,  0.4137],
 [ 0.9385,  0.4565,  0.7702],
 [ 0.4135, -0.2587,  0.0482]]])
data = torch.tensor([1., 1., 1.])
result = F.layer_norm(input, [3], data, data)

input = torch.tensor([[[ 1.1524,  0.4714,  0.2857],
 [-1.2533, -0.9829, -1.0981],
 [ 0.1507, -1.1431, -2.0361]],

[[ 0.1024, -0.4482,  0.4137],
 [ 0.9385,  0.4565,  0.7702],
 [ 0.4135, -0.2587,  0.0482]]])
data = torch.tensor([1., 1., 1.])
result = F.layer_norm(input=input, normalized_shape=[3], weight=data, bias=data)

# torch.nn.functional.leaky_relu
x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
result = F.leaky_relu(x)

x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
result = F.leaky_relu(x, 0.06)

# torch.nn.functional.leaky_relu_
x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
F.leaky_relu_(input=x, negative_slope=0.08)

x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
F.leaky_relu_(x, 0.06)

# torch.nn.functional.logsigmoid
x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
result = F.logsigmoid(x)

x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
result = F.logsigmoid(input=x)

# torch.nn.functional.lp_pool1d
input = torch.tensor([[[-0.5743, 0.4889, -0.0878, 0.4210, -0.0844],
                        [0.3614, 0.8458, -0.6152, 0.6894, 0.2927],
                        [-0.0087, 0.1098, 0.1783, -0.6953, 0.5519],
                        [0.3789, -0.0560, -0.4090, -0.1070, -1.0139],
                        [0.9204, 1.0817, -2.6126, 0.4244, 0.3272]]])
result = torch.nn.functional.lp_pool1d(input, 1, 2)

input = torch.tensor([[[0.6430, 0.4511, -1.6757, 1.7116],
                        [-0.2288, -0.4111, -1.3602, 0.2685],
                        [0.2363, 1.9341, 0.8522, -0.1846],
                        [1.6496, -0.0675, -0.7208, -1.0018]],

                        [[-0.3183, 0.8029, -0.4993, 1.0598],
                        [-0.4952, -0.9536, 0.1954, 0.0551],
                        [1.2257, 0.7517, 0.4063, -1.2151],
                        [-1.3562, 0.3547, 1.1147, 1.2898]],

                        [[0.1205, -0.1889, 0.5086, -0.8080],
                        [0.3156, -0.8298, 2.0242, -0.9184],
                        [-0.4005, 1.3586, 0.6205, -0.7487],
                        [1.6239, 0.2900, 0.9671, 1.2961]],

                        [[-1.1996, -0.2201, -0.9466, -0.7264],
                        [-0.0313, 0.8284, -0.3588, 1.3522],
                        [-0.0991, -0.5112, -0.1785, 2.0903],
                        [-1.3286, -0.9333, -0.1404, 1.2582]]])
result = torch.nn.functional.lp_pool1d(input, 4, 2, 2)

# torch.nn.functional.lp_pool2d
input = torch.tensor([[[[-0.5743,  0.4889, -0.0878,  0.4210, -0.0844],
    [ 0.3614,  0.8458, -0.6152,  0.6894,  0.2927],
    [-0.0087,  0.1098,  0.1783, -0.6953,  0.5519],
    [ 0.3789, -0.0560, -0.4090, -0.1070, -1.0139],
    [ 0.9204,  1.0817, -2.6126,  0.4244,  0.3272]]]])
result = torch.nn.functional.lp_pool2d(input, 1, 3, stride=2)

input = torch.tensor([[[[ 0.6430,  0.4511, -1.6757,  1.7116],
    [-0.2288, -0.4111, -1.3602,  0.2685],
    [ 0.2363,  1.9341,  0.8522, -0.1846],
    [ 1.6496, -0.0675, -0.7208, -1.0018]],

    [[-0.3183,  0.8029, -0.4993,  1.0598],
    [-0.4952, -0.9536,  0.1954,  0.0551],
    [ 1.2257,  0.7517,  0.4063, -1.2151],
    [-1.3562,  0.3547,  1.1147,  1.2898]],

    [[ 0.1205, -0.1889,  0.5086, -0.8080],
    [ 0.3156, -0.8298,  2.0242, -0.9184],
    [-0.4005,  1.3586,  0.6205, -0.7487],
    [ 1.6239,  0.2900,  0.9671,  1.2961]],

    [[-1.1996, -0.2201, -0.9466, -0.7264],
    [-0.0313,  0.8284, -0.3588,  1.3522],
    [-0.0991, -0.5112, -0.1785,  2.0903],
    [-1.3286, -0.9333, -0.1404,  1.2582]]]])
result = torch.nn.functional.lp_pool2d(input, 2, 4, stride=2)

# torch.nn.functional.max_pool1d
input = torch.tensor([[[ 1.1524,  0.4714,  0.2857],
    [-1.2533, -0.9829, -1.0981],
    [ 0.1507, -1.1431, -2.0361]]])
result = F.max_pool1d(input , 3)

input = torch.tensor([[[ 1.1524,  0.4714,  0.2857],
    [-1.2533, -0.9829, -1.0981],
    [ 0.1507, -1.1431, -2.0361]]])
result = F.max_pool1d(input , 3, stride=2)

# torch.nn.functional.max_pool2d
input = torch.tensor([[[[ 1.1524,  0.4714,  0.2857],
    [-1.2533, -0.9829, -1.0981],
    [ 0.1507, -1.1431, -2.0361]],

[[ 0.1024, -0.4482,  0.4137],
    [ 0.9385,  0.4565,  0.7702],
    [ 0.4135, -0.2587,  0.0482]]]])
result = F.max_pool2d(input , 3)

input = torch.tensor([[[[ 1.1524,  0.4714,  0.2857],
    [-1.2533, -0.9829, -1.0981],
    [ 0.1507, -1.1431, -2.0361]],

[[ 0.1024, -0.4482,  0.4137],
    [ 0.9385,  0.4565,  0.7702],
    [ 0.4135, -0.2587,  0.0482]]]])
result = F.max_pool2d(input , (3, 1))

# torch.nn.functional.max_pool3d
input = torch.arange(4800, dtype=torch.float32).reshape(2, 3, 8, 10, 10)
result = F.max_pool3d(input , 3)

input = torch.arange(4800, dtype=torch.float32).reshape(2, 3, 8, 10, 10)
result, indices = F.max_pool3d(input , 3, 1, 1, 2, True, True)

# torch.nn.functional.mish
x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
result = F.mish(x)

x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
result = F.mish(input=x)

# torch.nn.functional.normalize
x = torch.tensor([[[[-1.2392, -0.1310, -0.6679,  0.5476],
                    [ 1.1738, -1.7384, -0.7733,  0.3261],
                    [-0.0926, -1.0448, -1.2557, -1.5503],
                    [ 0.6402,  0.9072,  0.6780, -1.9885]],

                    [[ 0.0639, -1.1592,  1.4242, -0.4641],
                    [-0.1920,  0.1826,  1.9217, -0.4359],
                    [ 1.1926, -0.0247,  0.4744, -1.0216],
                    [-0.0360, -1.1656,  0.3661, -1.8147]]]])
result = F.normalize(x)

x = torch.tensor([[[[-1.2392, -0.1310, -0.6679,  0.5476],
                    [ 1.1738, -1.7384, -0.7733,  0.3261],
                    [-0.0926, -1.0448, -1.2557, -1.5503],
                    [ 0.6402,  0.9072,  0.6780, -1.9885]],

                    [[ 0.0639, -1.1592,  1.4242, -0.4641],
                    [-0.1920,  0.1826,  1.9217, -0.4359],
                    [ 1.1926, -0.0247,  0.4744, -1.0216],
                    [-0.0360, -1.1656,  0.3661, -1.8147]]]])
result = F.normalize(x, 3., 1)

# torch.nn.functional.one_hot
x = torch.tensor([1, 2, 0, 3, 5]) % 3
result = F.one_hot(x)

x = torch.tensor([1, 2, 0, 3, 5]) % 3
result = F.one_hot(x, 3)

# torch.nn.functional.pairwise_distance
a = [1.3192, 1.9915, 1.9674, 1.7151]
b = [1.3492, 0.1915, 2.9434, 1.4151]
x1 = torch.tensor(a)
x2 = torch.tensor(b)
result = torch.nn.functional.pairwise_distance(x1, x2, 2.0, 1e-6, False)

a = [1.3192, 1.9915, 1.9674, 1.7151]
b = [1.3492, 0.1915, 2.9434, 1.4151]
x1 = torch.tensor(a)
x2 = torch.tensor(b)
result = torch.nn.functional.pairwise_distance(x1=x1, x2=x2, p=1.0, eps=1e-6, keepdim=False)

# torch.nn.functional.relu
x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
result = F.relu(x)

x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
result = F.relu(x, False)

# torch.nn.functional.relu_
x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
result = F.relu_(x)

x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
result = F.relu_(input=x)

# torch.nn.functional.sigmoid
x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
result = F.sigmoid(x)

x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
result = F.sigmoid(input=x)

# torch.nn.functional.silu
x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
result = F.silu(x)

x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
result = F.silu(x, True)

# torch.nn.functional.softplus
x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
result = F.softplus(x)

x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
result = F.softplus(x, 3, 15)

# torch.nn.functional.softshrink
x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
result = F.softshrink(x)

x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
result = F.softshrink(x, 0.3)

# torch.nn.functional.tanh
x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
result = F.tanh(x)

x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
result = F.tanh(input=x)

# torch.nn.init._calculate_fan_in_and_fan_out
x = torch.randn(5, 5)
result = torch.nn.init._calculate_fan_in_and_fan_out(x)

x = torch.randn(5, 5, 5)
result = torch.nn.init._calculate_fan_in_and_fan_out(x)

# torch.nn.init.constant_
conv = torch.nn.Conv2d(4, 6, (3, 3))
torch.nn.init.constant_(conv.weight, val=0.2)
result = conv.weight

conv = torch.nn.Conv2d(3, 6, (3, 3))
torch.nn.init.constant_(conv.weight, val=2)
result = conv.weight

# torch.nn.init.dirac_
conv = torch.nn.Conv2d(4, 6, (3, 3))
torch.nn.init.dirac_(conv.weight)
result = conv.weight

conv = torch.nn.Conv2d(3, 6, (3, 3))
torch.nn.init.dirac_(conv.weight, 1)
result = conv.weight

# torch.nn.init.eye_
conv = torch.empty(3, 5)
torch.nn.init.eye_(conv)
result = conv

conv = torch.empty(3, 5)
torch.nn.init.eye_(tensor=conv)
result = conv

# torch.nn.init.normal_
conv = torch.nn.Conv2d(4, 6, (3, 3))
torch.nn.init.normal_(conv.weight)
result = conv.weight

conv = torch.nn.Conv2d(4, 6, (3, 3))
torch.nn.init.normal_(conv.weight, 0.2, 2.)
result = conv.weight

# torch.nn.init.ones_
conv = torch.nn.Conv2d(4, 6, (3, 3))
torch.nn.init.ones_(conv.weight)
result = conv.weight

conv = torch.nn.Conv2d(3, 6, (3, 3))
torch.nn.init.ones_(tensor=conv.weight)
result = conv.weight

# torch.nn.init.orthogonal_
conv = torch.nn.Conv2d(4, 6, (3, 3))
torch.nn.init.orthogonal_(conv.weight)
result = conv.weight

conv = torch.nn.Conv2d(3, 6, (3, 3))
torch.nn.init.orthogonal_(tensor=conv.weight)
result = conv.weight

# torch.nn.init.trunc_normal_
conv = torch.nn.Conv2d(4, 6, (3, 3))
torch.nn.init.trunc_normal_(conv.weight)
result = conv.weight

conv = torch.nn.Conv2d(3, 6, (3, 3))
torch.nn.init.trunc_normal_(tensor=conv.weight, mean=1., std=2., a=-1., b=1.)
result = conv.weight

# torch.nn.init.uniform_
conv = torch.nn.Conv2d(4, 6, (3, 3))
torch.nn.init.uniform_(conv.weight)
result = conv.weight

conv = torch.nn.Conv2d(3, 6, (3, 3))
torch.nn.init.uniform_(tensor=conv.weight)
result = conv.weight

# torch.nn.init.xavier_normal_
conv = torch.nn.Conv2d(4, 6, (3, 3))
torch.nn.init.xavier_normal_(conv.weight)
result = conv.weight

conv = torch.nn.Conv2d(3, 6, (3, 3))
torch.nn.init.xavier_normal_(tensor=conv.weight)
result = conv.weight

# torch.nn.init.xavier_uniform_
conv = torch.nn.Conv2d(4, 6, (3, 3))
torch.nn.init.xavier_uniform_(conv.weight)
result = conv.weight

conv = torch.nn.Conv2d(3, 6, (3, 3))
torch.nn.init.xavier_uniform_(tensor=conv.weight)
result = conv.weight

# torch.nn.init.zeros_
conv = torch.nn.Conv2d(4, 6, (3, 3))
torch.nn.init.zeros_(conv.weight)
result = conv.weight

conv = torch.nn.Conv2d(3, 6, (3, 3))
torch.nn.init.zeros_(tensor=conv.weight)
result = conv.weight

# torch.nn.parameter.Parameter
x = torch.tensor([[1., 2., 3.], [2., 3., 4.]])
result = torch.nn.parameter.Parameter(x)

x = torch.tensor([[1., 2., 3.], [2., 3., 4.]])
result = torch.nn.parameter.Parameter(x, requires_grad=False)

# torch.nn.utils.parameters_to_vector
model = nn.Linear(10, 20)
result = nn.utils.parameters_to_vector(model.parameters())

model = nn.Linear(10, 20)
result = nn.utils.parameters_to_vector(parameters=model.parameters())

# torch.nn.utils.rnn.pad_sequence
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0])
c = torch.tensor([6.0])
result = pad_sequence([a, b, c])

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0])
c = torch.tensor([6.0])
result = pad_sequence([a, b, c], batch_first=True)

# torch.nn.utils.rnn.unpad_sequence
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

# torch.nn.utils.vector_to_parameters
model = nn.Linear(10, 20)
a = nn.utils.parameters_to_vector(model.parameters())
b = nn.utils.vector_to_parameters(a, model.parameters())
result = a.detach()

model = nn.Linear(10, 20)
a = nn.utils.parameters_to_vector(model.parameters())
b = nn.utils.vector_to_parameters(vec=a, parameters=model.parameters())
result = a.detach()

# torch.no_grad
x = torch.tensor([1.], requires_grad=True)
with torch.no_grad():
    y = x * 2

@torch.no_grad()
def doubler(x):
    return x * 2
x = torch.tensor([1.], requires_grad=True)
y = doubler(x)

# torch.nonzero
result = torch.nonzero(torch.tensor([1, 1, 1, 0, 1]))

result = torch.nonzero(torch.tensor([[0.6, 0.0, 0.0, 0.0],
                            [0.0, 0.4, 0.0, 0.0],
                            [0.0, 0.0, 1.2, 0.0],
                            [0.0, 0.0, 0.0,-0.4]]))

# torch.norm
input = torch.tensor([[[-12., -11., -10., -9. ],
                    [-8. , -7. , -6. , -5. ],
                    [-4. , -3. , -2. , -1. ]],
                    [[ 0. ,  1. ,  2. ,  3. ],
                    [ 4. ,  5. ,  6. ,  7. ],
                    [ 8. ,  9. ,  10.,  11.]]])
result = torch.norm(input, p='fro')

input = torch.tensor([[-12., -11., -10., -9. ],
            [-8. , -7. , -6. , -5. ],
            [-4. , -3. , -2. , -1. ]])
result = torch.norm(input, p='nuc')

# torch.normal
result = torch.normal(torch.arange(1., 11.), torch.arange(1, 11))

result = torch.normal(mean=0.5, std=torch.arange(1., 6.))

# torch.not_equal
result = torch.not_equal(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))

input = torch.tensor([[1, 2], [3, 4]])
other = torch.tensor([[1, 1], [4, 4]])
result = torch.not_equal(input, other)

# torch.ones
result = torch.ones(3)

result = torch.ones(3, 5)

# torch.ones_like
input = torch.empty(2, 3)
result = torch.ones_like(input)

result = torch.ones_like(torch.empty(2, 3))

# torch.outer
x = torch.tensor([1., 2, 3])
y = torch.tensor([1., 2, 3, 4])
result = torch.outer(x, y)

x = torch.tensor([1., 2, 3])
y = torch.tensor([1., 2, 3, 4])
result = torch.outer(input=x, vec2=y)

# torch.permute
x = torch.tensor([[1, 2, 3], [3, 4, 6]])
result = torch.permute(x, (1, 0))

x = torch.tensor([[1, 2, 3], [3, 4, 6]])
result = torch.permute(x, [1, 0])

# torch.pi
result = torch.pi

# torch.polar
abs = torch.tensor([1, 2], dtype=torch.float64)
angle = torch.tensor([np.pi / 2, 5 * np.pi / 4], dtype=torch.float64)
result = torch.polar(abs, angle)

abs = torch.tensor([1, 2], dtype=torch.float64)
angle = torch.tensor([np.pi / 2, 5 * np.pi / 4], dtype=torch.float64)
out = torch.tensor([1, 2], dtype=torch.complex128)
result = torch.polar(abs, angle, out=out)

# torch.pow
a = torch.tensor([0.4331,  1.2475,  0.6834, -0.2791])
result = torch.pow(a, 2)

a = torch.tensor([0.4331,  1.2475,  0.6834, -0.2791])
b = torch.tensor([1, 2, 3, 4])
result = torch.pow(a, b)

# torch.prod
input = torch.tensor([1.4907, 1.0593, 1.5696])
result = torch.prod(input)

input = torch.tensor([[1.4907, 1.0593, 1.5696], [1.4907, 1.0593, 1.5696]])
result = torch.prod(input, 1)

# torch.quantile
result = torch.quantile(torch.tensor([0., 1., 2., 3.],dtype=torch.float64), 0.6)

result = torch.quantile(torch.tensor([0., 1., 2., 3.],dtype=torch.float64), 0.6, dim=None)

# torch.rand
result = torch.rand(3)

result = torch.rand(3, 5)

# torch.rand_like
a = torch.zeros(3, 4, dtype=torch.float64)
result = torch.rand_like(a)

a = torch.zeros(3, 4, dtype=torch.float64)
result = torch.rand_like(a, dtype=torch.float32, requires_grad=True)

# torch.randn
result = torch.randn(3)

result = torch.randn(3, 5)

# torch.randn_like
a = torch.zeros(3, 4, dtype=torch.float64)
result = torch.randn_like(a)

a = torch.zeros(3, 4, dtype=torch.float64)
result = torch.randn_like(a, dtype=torch.float32, requires_grad=True)

# torch.random.initial_seed
torch.manual_seed(100)
result = torch.random.initial_seed()

# torch.randperm
result = torch.randperm(5)

n = 5
result = torch.randperm(n)

# torch.range
result = torch.range(1,4)

result = torch.range(1,4,0.5)

# torch.ravel
a = torch.tensor([[4, 9], [23, 2]])
result = torch.ravel(a)

result = torch.ravel(torch.tensor([[4, 9], [23, 2]]))

# torch.reciprocal
result = torch.reciprocal(torch.tensor([-0.4595, -2.1219, -1.4314,  0.7298]))

a = torch.tensor([-0.4595, -2.1219, -1.4314,  0.7298])
result = torch.reciprocal(a)

# torch.remainder
a = torch.tensor([-3., -2, -1, 1, 2, 3])
result = torch.remainder(a, 2.)

a = torch.tensor([1, 2, 3, 4, 5])
result = torch.remainder(a, torch.tensor(1.5))

# torch.renorm
x = torch.tensor([[ 1.,  1.,  1.],
                    [ 2.,  2.,  2.],
                    [ 3.,  3.,  3.]])
result = torch.renorm(x, 1, 0, 5)

x = torch.tensor([[ 1.,  1.,  1.],
                    [ 2.,  2.,  2.],
                    [ 3.,  3.,  3.]])
result = torch.renorm(input=x, p=1, dim=0, maxnorm=5)

# torch.repeat_interleave
a = torch.tensor([[4, 9], [23, 2]])
result = torch.repeat_interleave(a, 3, 0)

a = torch.tensor([[4, 9], [23, 2]])
result = torch.repeat_interleave(input=a, repeats=3, dim=1)

# torch.reshape
a = torch.arange(4.)
result = torch.reshape(a, (2, 2))

a = torch.arange(9)
shape = (3, 3)
result = torch.reshape(a, shape)

# torch.roll
x = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
result = torch.roll(x, 1)

x = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
result = torch.roll(x, 1, 0)

# torch.round
a = torch.tensor([[[ 0.9254, -0.6213],
    [-0.5787,  1.6843]],

    [[ 0.3242, -0.9665],
    [ 0.4539, -0.0887]],

    [[ 1.1336, -0.4025],
    [-0.7089,  0.9032]]])
result = torch.round(a)

result = torch.round(torch.tensor([[[ 0.9254, -0.6213],
    [-0.5787,  1.6843]],

    [[ 0.3242, -0.9665],
    [ 0.4539, -0.0887]],

    [[ 1.1336, -0.4025],
    [-0.7089,  0.9032]]]))

# torch.rsqrt
result = torch.rsqrt(torch.tensor([0.2970,  1.5420, 4]))

a = torch.tensor([0.2970,  1.5420, 4])
result = torch.rsqrt(a)

# torch.scatter
input = torch.arange(15).reshape([3, 5]).type(torch.float32)
index = torch.tensor([[0], [1], [2]])
result = torch.scatter(input, 1, index, 1.0)

input = torch.arange(15).reshape([3, 5]).type(torch.float32)
index = torch.tensor([[0], [1], [2]])
result = torch.scatter(input=input, dim=1, index=index, value=1.0)

# torch.scatter_add
src = torch.ones((1, 5))
index = torch.tensor([[0, 1, 2, 0, 0]])
input = torch.zeros(3, 5, dtype=src.dtype)
result = torch.scatter_add(input,0, index, src)

src = torch.ones((2, 5))
index = torch.tensor([[0, 1, 2, 0, 0], [0, 1, 2, 2, 2]])
input = torch.zeros(3, 5, dtype=src.dtype)
result = torch.scatter_add(input,0, index, src)

# torch.scatter_reduce
src = torch.tensor([1., 2., 3., 4., 5., 6.])
index = torch.tensor([0, 1, 0, 1, 2, 1])
input = torch.tensor([1., 2., 3., 4.])
type = "sum"
result = torch.scatter_reduce(input, 0, index, src, reduce=type)

src = torch.tensor([1., 2., 3., 4., 5., 6.])
index = torch.tensor([0, 1, 0, 1, 2, 1])
input = torch.tensor([1., 2., 3., 4.])
result = torch.scatter_reduce(input=input, dim=0, index=index, src=src, reduce="sum", include_self=False)

# torch.searchsorted
x = torch.tensor([[ 1,  3,  5,  7,  9],
                  [ 2,  4,  6,  8, 10]])
values = torch.tensor([[3, 6, 9],
                       [3, 6, 9]])
result = torch.searchsorted(x, values)

x = torch.tensor([[ 1,  3,  5,  7,  9],
                  [ 2,  4,  6,  8, 10]])
values = torch.tensor([[3, 6, 9],
                       [3, 6, 9]])
result = torch.searchsorted(x, values, out_int32 = True)

# torch.set_default_dtype
torch.set_default_dtype(torch.float64)
result = torch.tensor([1.2, 3])

torch.set_default_dtype(torch.float64)
result = torch.tensor([1.2, 3j])

# torch.sigmoid
x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
result = torch.sigmoid(x)

x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
result = torch.sigmoid(input=x)

# torch.sign
result = torch.sign(torch.tensor([ 0.9213,  1.0887, -0.8858, -1.7683]))

a = torch.tensor([ 0.9213,  1.0887, -0.8858, -1.7683])
result = torch.sign(a)

# torch.sin
result = torch.sin(torch.tensor([1.4309,  1.2706, -0.8562,  0.9796]))

a = torch.tensor([ 1.4309,  1.2706, -0.8562,  0.9796])
result = torch.sin(a)

# torch.sinh
result = torch.sinh(torch.tensor([1.4309,  1.2706, -0.8562,  0.9796]))

a = torch.tensor([ 1.4309,  1.2706, -0.8562,  0.9796])
result = torch.sinh(a)

# torch.softmax
input = torch.tensor([[-1.2837, -0.0297,  0.0355],
    [ 0.9112, -1.7526, -0.4061]])
result = torch.softmax(input, dim=0)

input = torch.tensor([[-1.2837, -0.0297,  0.0355],
    [ 0.9112, -1.7526, -0.4061]])
result = torch.softmax(input, dim=1)

# torch.special.expm1
result = torch.special.expm1(torch.tensor([0., -2., 3.]))

a = torch.tensor([-1., -2., 3.])
result = torch.special.expm1(a)

# torch.special.i0
result = torch.special.i0(torch.tensor([1.0, 2.0, 3.0]))

x = torch.tensor([1.0, 2.0, 3.0])
result = torch.special.i0(x)

# torch.special.i0e
x = torch.tensor([1.0, 2.0, 3.0])
result = torch.special.i0e(x)

x = torch.tensor([1.0, 2.0, 3.0])
result = torch.special.i0e(input=x)

# torch.special.i1
result = torch.special.i1(torch.tensor([1.0, 2.0, 3.0]))

x = torch.tensor([1.0, 2.0, 3.0])
result = torch.special.i1(x)

# torch.special.i1e
result = torch.special.i1e(torch.tensor([1.0, 2.0, 3.0]))

x = torch.tensor([1.0, 2.0, 3.0])
result = torch.special.i1e(x)

# torch.special.logsumexp
input = torch.tensor([1.4907, 1.0593, 1.5696])
result = torch.special.logsumexp(input, 0)

input = torch.tensor([[1.4907, 1.0593, 1.5696], [1.4907, 1.0593, 1.5696]])
result = torch.special.logsumexp(input, 1)

# torch.special.softmax
x = torch.tensor([[[2.0, 3.0, 4.0, 5.0],
                [3.0, 4.0, 5.0, 6.0],
                [7.0, 8.0, 8.0, 9.0]],
                [[1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [6.0, 7.0, 8.0, 9.0]]])
result = torch.special.softmax(x, -1)

x = torch.tensor([[[2.0, 3.0, 4.0, 5.0],
                [3.0, 4.0, 5.0, 6.0],
                [7.0, 8.0, 8.0, 9.0]],
                [[1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [6.0, 7.0, 8.0, 9.0]]])
result = torch.special.softmax(x, dim=1)

# torch.sqrt
result = torch.sqrt(torch.tensor([0.2970,  1.5420, 4]))

a = torch.tensor([0.2970,  1.5420, 4])
result = torch.sqrt(a)

# torch.square
result = torch.square(torch.tensor([0.2970,  1.5420, 4]))

a = torch.tensor([0.2970,  1.5420, 4])
result = torch.square(a)

# torch.squeeze
x = torch.zeros(2, 1, 2, 1, 2)
result = torch.squeeze(x)

result = torch.squeeze(torch.zeros(2, 1, 2, 1, 2))

# torch.stack
x = torch.zeros(2, 3)
y = torch.zeros(2, 3)
result = torch.stack((x, y))

result = torch.stack((torch.zeros(2, 3), torch.zeros(2, 3)))

# torch.std
input = torch.tensor([1.4907, 1.0593, 1.5696])
result = torch.std(input, unbiased=False)

input = torch.tensor([[1.4907, 1.0593, 1.5696], [1.4907, 1.0593, 1.5696]])
result = torch.std(input, unbiased=False)

# torch.sub
a = torch.tensor([ 0.5950,-0.0872, 2.3298, -0.2972])
b = torch.tensor([1, 1, 1, 0])
result = torch.sub(a, b)

a = torch.tensor([ 0.5950,-0.0872, 2.3298, -0.2972])
b = torch.tensor([1, 1, 1, 0])
result = torch.sub(input=a, other=b)

# torch.sum
input = torch.tensor([1.4907, 1.0593, 1.5696])
result = torch.sum(input)

input = torch.tensor([[1.4907, 1.0593, 1.5696], [1.4907, 1.0593, 1.5696]])
result = torch.sum(input, 1)

# torch.swapaxes
x = torch.tensor([[[0,1],[2,3]],[[4,5],[6,7]]])
result = torch.swapaxes(x, 0, 1)

x = torch.tensor([[[0,1],[2,3]],[[4,5],[6,7]]])
result = torch.swapaxes(input=x, axis0=0, axis1=1)

# torch.swapdims
x = torch.tensor([[[0,1],[2,3]],[[4,5],[6,7]]])
result = torch.swapdims(x, 0, 1)

x = torch.tensor([[[0,1],[2,3]],[[4,5],[6,7]]])
result = torch.swapdims(input=x, dim0=0, dim1=1)

# torch.t
x = torch.zeros(2, 3)
result = torch.t(x)

x = torch.zeros(2)
result = torch.t(x)

# torch.take_along_dim
x = torch.tensor([[1, 2, 3], [3, 4, 6]])
idx = torch.tensor([[0]])
result = torch.take_along_dim(x, idx, 1)

x = torch.tensor([[1, 2, 3], [3, 4, 6]])
idx = torch.tensor([[0]])
result = torch.take_along_dim(input=x, indices=idx, dim=0)

# torch.tan
result = torch.tan(torch.tensor([1.4309,  1.2706, -0.8562,  0.9796]))

a = torch.tensor([ 1.4309,  1.2706, -0.8562,  0.9796])
result = torch.tan(a)

# torch.tanh
x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
result = torch.tanh(x)

x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                    [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
result = torch.tanh(input=x)

# torch.tensor
result = torch.tensor([2, 3])

data = [2, 3]
result = torch.tensor(data)

# torch.tensor_split
a = torch.arange(8)
result = torch.tensor_split(a, 3)

a = torch.arange(7)
result = torch.tensor_split(a, sections = 3)

# torch.testing.assert_close
x = torch.tensor([1., 2., 3.])
torch.testing.assert_close(x, x)

x = torch.tensor([1., 2., 3.])
torch.testing.assert_close(actual=x, expected=x)

# torch.tile
x = torch.tensor([1, 2, 3])
result = torch.tile(x, (2,))

x = torch.tensor([[1, 2], [0, 6]])
result = torch.tile(x, (2, 3))

# torch.topk
x = torch.tensor([1, 2, 3, 4, 5])
result, index = torch.topk(x, 3)

x = torch.tensor([1, 2, 3, 4, 5])
res = torch.topk(x, 3)
result, index = res[0], res[1]

# torch.torch.int32
src = torch.tensor([1., 2., 3., 4., 5., 6.])
result = src.to(torch.torch.int32)

result = torch.tensor([1., 2., 3., 4., 5., 6.]).to(torch.torch.int32)

# torch.transpose
a = torch.Tensor([[1.,2.], [3.,4.]])
result = torch.transpose(a, dim0=0, dim1=1)

a = torch.Tensor([[1.,2.], [3.,4.]])
result = torch.transpose(a, 0, 1)

# torch.tril
x = torch.tensor([[-1.0813, -0.8619,  0.7105],
                [ 0.0935,  0.1380,  2.2112],
                [-0.3409, -0.9828,  0.0289]])
result = torch.tril(x)

x = torch.tensor([[-1.0813, -0.8619,  0.7105],
                [ 0.0935,  0.1380,  2.2112],
                [-0.3409, -0.9828,  0.0289]])
result = torch.tril(x, 1)

# torch.triu
x = torch.tensor([[-1.0813, -0.8619,  0.7105],
                [ 0.0935,  0.1380,  2.2112],
                [-0.3409, -0.9828,  0.0289]])
result = torch.triu(x)

x = torch.tensor([[-1.0813, -0.8619,  0.7105],
                [ 0.0935,  0.1380,  2.2112],
                [-0.3409, -0.9828,  0.0289]])
result = torch.triu(x, 1)

# torch.true_divide
a = torch.tensor([4.67, 9.76 , 8.53])
b = torch.tensor([3.5, 3.90, 1.83])
result = torch.true_divide(a, b)

a = torch.tensor([[4., 9., 8.]])
b = torch.tensor([2., 3., 4.])
result = torch.true_divide(a, b)

# torch.uint8
src = torch.tensor([1., 2., 3., 4., 5., 6.])
result = src.to(torch.uint8)

result = torch.tensor([1., 2., 3., 4., 5., 6.]).to(torch.uint8)

# torch.unbind
x = torch.tensor([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])
result = torch.unbind(x)

x = torch.tensor([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])
result = torch.unbind(x, 1)

# torch.unflatten
a = torch.tensor([[ 0.2180,  1.0558,  0.1608,  0.9245],
        [ 1.3794,  1.4090,  0.2514, -0.8818],
        [-0.4561,  0.5123,  1.7505, -0.4094]])
result = torch.unflatten(a, -1, (2, 2))

a = torch.tensor([[ 0.2180,  1.0558,  0.1608,  0.9245],
        [ 1.3794,  1.4090,  0.2514, -0.8818],
        [-0.4561,  0.5123,  1.7505, -0.4094]])
result = torch.unflatten(a, 1, (2, 2))

# torch.unique_consecutive
x = torch.tensor([1, 1, 2, 2, 3, 1, 1, 2])
result = torch.unique_consecutive(x)

x = torch.tensor([1, 1, 2, 2, 3, 1, 1, 2])
result, inverse_indices = torch.unique_consecutive(x, return_inverse=True)

# torch.unsqueeze
x = torch.zeros(2, 2, 2)
result = torch.unsqueeze(x, 0)

result = torch.unsqueeze(torch.zeros(2, 2, 1, 2), 3)

# torch.utils.cpp_extension.BuildExtension
CppExtension(
    name='extension',
    sources=['extension.cpp'],
    extra_compile_args=['-g'])
dic = {'build_ext': BuildExtension}
result = True

CppExtension(
    'extension',
    ['extension.cpp'],
    extra_compile_args=['-g'])
dic = {'build_ext': BuildExtension}
result = True

# torch.utils.cpp_extension.BuildExtension.with_options
CppExtension(
    name='extension',
    sources=['extension.cpp'],
    extra_compile_args=['-g'])
dic = {'build_ext': BuildExtension.with_options}
result = True

# torch.utils.cpp_extension.CUDA_HOME
result = CUDA_HOME

# torch.utils.data.ChainDataset
class MyIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, start, end):
        super(MyIterableDataset).__init__()
        assert end > start, "this example code only works with end >= start"
        self.start = start
        self.end = end

    def __iter__(self):
        iter_start = self.start
        iter_end = self.end
        return iter(range(iter_start, iter_end))


dataset = torch.utils.data.ChainDataset([MyIterableDataset(start=3, end=7), MyIterableDataset(start=3, end=7)])
result = []
for d in dataset:
    result.append(d)

class MyIterableDataset(torch_data.IterableDataset):
    def __init__(self, start, end):
        super(MyIterableDataset).__init__()
        assert end > start, "this example code only works with end >= start"
        self.start = start
        self.end = end

    def __iter__(self):
        iter_start = self.start
        iter_end = self.end
        return iter(range(iter_start, iter_end))


dataset = torch_data.ChainDataset([MyIterableDataset(start=1, end=10), MyIterableDataset(start=1, end=3)])
result = []
for d in dataset:
    result.append(d)

# torch.utils.data.ConcatDataset
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

class RandomDataset(torch_data.Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __getitem__(self, idx):
        image = np.arange(5).astype('float32')
        label = np.array([idx]).astype('int64')
        return torch.tensor(image), torch.tensor(label)

    def __len__(self):
        return self.num_samples

dataset = torch_data.ConcatDataset(datasets=[RandomDataset(2), RandomDataset(2)])
result = []
for i in range(len(dataset)):
    result.append(dataset[i])

# torch.utils.data.Dataset
class Data(Dataset):
    def __init__(self):
        self.x = [1,2,3,4]

    def __getitem__(self, idx):
        return self.x[idx]

    def __len__(self):
        return len(self.x)


data = Data()
result = data.__len__()

class Data(torch_data.Dataset):
    def __init__(self):
        self.x = [1,2,3,4]

    def __getitem__(self, idx):
        return self.x[idx]

    def __len__(self):
        return len(self.x)


my_data = Data()
result = my_data.__getitem__(0)

# torch.utils.data.IterableDataset
class MyIterableDataset(IterableDataset):
    def __init__(self, start, end):
        super(MyIterableDataset).__init__()
        assert end > start, "this example code only works with end >= start"
        self.start = start
        self.end = end

    def __iter__(self):
        return iter(range(self.start, self.end))

ds = MyIterableDataset(start=3, end=7)
result = []
for i in ds:
    result.append(i)

class MyIterableDataset(torch_data.IterableDataset):
    def __init__(self, start, end):
        super(MyIterableDataset).__init__()
        assert end > start, "this example code only works with end >= start"
        self.start = start
        self.end = end

    def __iter__(self):
        return iter(range(self.start, self.end))

ds = MyIterableDataset(start=3, end=7)
result = next(ds.__iter__())

# torch.utils.data.Sampler
class Data(Dataset):
    def __init__(self):
        self.x = np.arange(0,100,1)

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

class Data(torch_data.Dataset):
    def __init__(self):
        self.x = np.arange(0,100,1)

    def __getitem__(self, idx):
        return self.x[idx]

    def __len__(self):
        return self.x.shape[0]

class MySampler(torch_data.Sampler):
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

# torch.utils.data.SequentialSampler
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

class MyDataset(torch_data.Dataset):
    def __init__(self):
        self.x = np.arange(0, 100, 1)

    def __getitem__(self, idx):
        return self.x[idx]

    def __len__(self):
        return len(self.x)

my_data = MyDataset()
s = torch_data.SequentialSampler(data_source=my_data)
result = []
for d in s:
    result.append(d)

# torch.utils.data.get_worker_info
result = torch.utils.data.get_worker_info()

class MyIterableDataset(IterableDataset):
    def __init__(self, start, end):
        super(MyIterableDataset).__init__()
        assert end > start, "this example code only works with end >= start"
        self.start = start
        self.end = end

    def __iter__(self):
        worker_info = get_worker_info()
        iter_start = self.start
        iter_end = self.end
        return iter(range(iter_start, iter_end))


dataset = ChainDataset([MyIterableDataset(start=1, end=10), MyIterableDataset(start=1, end=3)])
result = []
for d in dataset:
    result.append(d)

# torch.utils.data.random_split
class Data(torch.utils.data.Dataset):
    def __init__(self):
        self.x = [0,1,2,3,4,5,6,7,8,9]

    def __getitem__(self, idx):
        return self.x[idx]

    def __len__(self):
        return len(self.x)


data = Data()

datasets = torch.utils.data.random_split(data, [3, 7])

results = []
for d in datasets:
    results.append(d.__len__())

class Data(torch_data.Dataset):
    def __init__(self):
        self.x = [0,1,2,3,4,5,6,7,8,9]

    def __getitem__(self, idx):
        return self.x[idx]

    def __len__(self):
        return len(self.x)


my_data = Data()
datasets = torch_data.random_split(my_data, [3, 3, 4])

results = []
for d in datasets:
    results.append(d.__len__())

# torch.var
input = torch.tensor([1.4907, 1.0593, 1.5696])
result = torch.var(input, unbiased=False)

input = torch.tensor([[1.4907, 1.0593, 1.5696], [1.4907, 1.0593, 1.5696]])
result = torch.var(input, unbiased=False)

# torch.view_as_complex
x = torch.tensor([[ 1.6116, -0.5772], [-1.4606, -0.9120]])
result = torch.view_as_complex(x)

x = torch.tensor([[ 1.6116, -0.5772], [-1.4606, -0.9120]])
result = torch.view_as_complex(input=x)

# torch.view_as_real
x = torch.tensor([(0.4737-0.3839j), (-0.2098-0.6699j), (0.3470-0.9451j), (-0.5174-1.3136j)])
result = torch.view_as_real(x)

x = torch.tensor([(0.4737-0.3839j), (-0.2098-0.6699j), (0.3470-0.9451j), (-0.5174-1.3136j)])
result = torch.view_as_real(input=x)

# torch.where
x = torch.tensor([[0.9383, -0.1983, 3.2, -1.2]])
y = 10.
result = torch.where(x>0, x, y)

x = torch.tensor([[0.9383, -0.1983, 3.2, -1.2]])
y = 10
result = torch.where(x>0, x, y)

# torch.zeros
result = torch.zeros(3)

result = torch.zeros(3, 5)

# torch.zeros_like
input = torch.empty(2, 3)
result = torch.zeros_like(input)

result = torch.zeros_like(torch.empty(2, 3))

# torch.Tensor.bfloat16
src = torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.uint8)
result = src.bfloat16().float()

src = torch.tensor([0.+3.5j, -1+4.2j, 2.34-5.2j, -3.45+7.9j, -0.34-8.2j, 0.23+9.2j, 1.+1.j, 2.+0.5j, 3.-1.j,], dtype=torch.complex64)
result = src.bfloat16().float()

# torch.Tensor.bool
src = torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.uint8)
result = src.bool()

src = torch.tensor([0.+3.5j, -1+4.2j, 2.34-5.2j, -3.45+7.9j, -0.34-8.2j, 0.23+9.2j, 1.+1.j, 2.+0.5j, 3.-1.j,], dtype=torch.complex64)
result = src.bool()

# torch.Tensor.byte
src = torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.uint8)
result = src.byte()

src = torch.tensor([0.+3.5j, -1+4.2j, 2.34-5.2j, -3.45+7.9j, -0.34-8.2j, 0.23+9.2j, 1.+1.j, 2.+0.5j, 3.-1.j,], dtype=torch.complex64)
result = src.byte()

# torch.Tensor.cdouble
src = torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.uint8)
result = src.cdouble()

src = torch.tensor([0.+3.5j, -1+4.2j, 2.34-5.2j, -3.45+7.9j, -0.34-8.2j, 0.23+9.2j, 1.+1.j, 2.+0.5j, 3.-1.j,], dtype=torch.complex64)
result = src.cdouble()

# torch.Tensor.cfloat
src = torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.uint8)
result = src.cfloat()

src = torch.tensor([0.+3.5j, -1+4.2j, 2.34-5.2j, -3.45+7.9j, -0.34-8.2j, 0.23+9.2j, 1.+1.j, 2.+0.5j, 3.-1.j,], dtype=torch.complex64)
result = src.cfloat()

# torch.Tensor.char
src = torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.uint8)
result = src.char()

src = torch.tensor([0.+3.5j, -1+4.2j, 2.34-5.2j, -3.45+7.9j, -0.34-8.2j, 0.23+9.2j, 1.+1.j, 2.+0.5j, 3.-1.j,], dtype=torch.complex64)
result = src.char()

# torch.Tensor.double
src = torch.tensor([1., 2., 3., 4., 5., 6.])
result = src.double()

src = torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.uint8)
result = src.double()

# torch.Tensor.float
src = torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.uint8)
result = src.float()

src = torch.tensor([0.+3.5j, -1+4.2j, 2.34-5.2j, -3.45+7.9j, -0.34-8.2j, 0.23+9.2j, 1.+1.j, 2.+0.5j, 3.-1.j,], dtype=torch.complex64)
result = src.float()

# torch.Tensor.half
src = torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.uint8)
result = src.half()

src = torch.tensor([0.+3.5j, -1+4.2j, 2.34-5.2j, -3.45+7.9j, -0.34-8.2j, 0.23+9.2j, 1.+1.j, 2.+0.5j, 3.-1.j,], dtype=torch.complex64)
result = src.half()

# torch.Tensor.int
src = torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.uint8)
result = src.int()

src = torch.tensor([0.+3.5j, -1+4.2j, 2.34-5.2j, -3.45+7.9j, -0.34-8.2j, 0.23+9.2j, 1.+1.j, 2.+0.5j, 3.-1.j,], dtype=torch.complex64)
result = src.int()

# torch.Tensor.long
src = torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.uint8)
result = src.long()

src = torch.tensor([0.+3.5j, -1+4.2j, 2.34-5.2j, -3.45+7.9j, -0.34-8.2j, 0.23+9.2j, 1.+1.j, 2.+0.5j, 3.-1.j,], dtype=torch.complex64)
result = src.long()

# torch.Tensor.short
src = torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.uint8)
result = src.short()

src = torch.tensor([0.+3.5j, -1+4.2j, 2.34-5.2j, -3.45+7.9j, -0.34-8.2j, 0.23+9.2j, 1.+1.j, 2.+0.5j, 3.-1.j,], dtype=torch.complex64)
result = src.short()

# torch.nn.init.calculate_gain
result = torch.nn.init.calculate_gain('leaky_relu', 0.2)
result = torch.tensor(result)

result = torch.nn.init.calculate_gain(nonlinearity='relu', param=0.2)
result = torch.tensor(result)

# torch.nn.init.kaiming_normal_
conv = torch.nn.Conv2d(4, 6, (3, 3))
torch.nn.init.kaiming_normal_(conv.weight)
result = conv.weight

conv = torch.nn.Conv2d(4, 6, (3, 3))
torch.nn.init.kaiming_normal_(conv.weight)
result = conv.weight

# torch.nn.init.kaiming_uniform_
conv = torch.nn.Conv2d(4, 6, (3, 3))
torch.nn.init.kaiming_uniform_(conv.weight)
result = conv.weight

conv = torch.nn.Conv2d(4, 6, (3, 3))
torch.nn.init.kaiming_uniform_(conv.weight)
result = conv.weight

# torch.utils.data.Subset
class MyDataset(Dataset):
    def __init__(self, size=10):
        super(Dataset).__init__()
        self.data = list(range(size))

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

dataset = Subset(MyDataset(10),[1, 2, 3, 4, 5, 6])
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

# torch.Tensor.bernoulli
a = torch.tensor([0.8, 0.1, 0.4])
result = a.bernoulli()

a = torch.ones(3, 3)
result = a.bernoulli()

# torch.Tensor.count_nonzero
a = torch.tensor([[0., 1.1, 1.2], [0., 0., 1.3], [0., 0., 0.]])
result = a.count_nonzero()

a = torch.tensor([[0., 1.1, 1.2], [0., 0., 1.3], [0., 0., 0.]])
result = a.count_nonzero(dim=1)

# torch.Tensor.floor_divide_
a = torch.tensor([4.0, 3.0])
b = torch.tensor([2.0, 2.0])
a.floor_divide_(b)

a = torch.tensor([4.0, 3.0])
b = torch.tensor([2.0, 2.0])
a.floor_divide_(other=b)

# torch.Tensor.hypot
a = torch.tensor([1., 2, 3])
b = torch.tensor([4., 5, 6])
result = a.hypot(b)

a = torch.tensor([1.])
b = torch.tensor([4., 5, 6])
result = a.hypot(other=b)

# torch.Tensor.kthvalue
x = torch.tensor([1., 2., 3., 4., 5.])
result = x.kthvalue(4)

x = torch.tensor([[ 1., 2., 3.], [ 4., 5., 6.]])
result = x.kthvalue(2, 0, True)

# torch.Tensor.logcumsumexp
x = torch.tensor([[0.56, 0.34, 0.78], [0.98, 0.21, 1.78]])
result = x.logcumsumexp(0)

x = torch.tensor([[0.56, 0.34, 0.78], [0.98, 0.21, 1.78]])
result = x.logcumsumexp(1)

# torch.Tensor.mode
input = torch.tensor([[[1,2,2],[2,3,3]],[[0,5,5],[9,9,0]]])
result, index = input.mode()

input = torch.tensor([[[1,2,2],[2,3,3]],[[0,5,5],[9,9,0]]])
result = input.mode()
result, index = result[0], result[1]

# torch.Tensor.pow_
a = torch.tensor([0.4331,  1.2475,  0.6834, -0.2791])
a.pow_(2)

a = torch.tensor([0.4331,  1.2475,  0.6834, -0.2791])
a.pow_(exponent=3.)

# torch.Tensor.remainder_
a = torch.tensor([-3., -2, -1, 1, 2, 3])
result = a.remainder_(torch.tensor(2.))

result = torch.tensor([-3., -2, -1, 1, 2, 3]).remainder_(torch.tensor(2.))

# torch.Tensor.split_with_sizes
a = torch.arange(10)
split_sizes = [3, 2, 5]
result = a.split_with_sizes(split_sizes, dim=0)

a = torch.arange(6).reshape(2, 3)
split_sizes = [1, 2]
result = a.split_with_sizes(split_sizes, dim=1)

# torch.Tensor.squeeze_
result = torch.zeros(2, 1, 2, 1, 2)
result.squeeze_()

result = torch.zeros(2, 1, 2, 1, 2).squeeze_()

# torch.Tensor.unsqueeze_
result = torch.zeros(2, 2, 2)
result.unsqueeze_(0)

result = torch.zeros(2, 2, 1, 2).unsqueeze_(3)

# torch.autograd.function.FunctionCtx
# Inherit from Function
class MyFunction(Function):
    @staticmethod
    def forward(ctx, x):
        # Store some tensors for backward
        ctx.x = x
        return x * 2

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve stored tensors
        x = ctx.x
        grad_input = grad_output * 2
        return grad_input

data = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
output = MyFunction.apply(data)
output.backward(torch.tensor([1.0, 1.0, 1.0]))

result = data.grad
result.requires_grad = False

# torch.autograd.function.FunctionCtx.mark_non_differentiable
# Inherit from Function
class cus_func(Function):
    @staticmethod
    def forward(ctx, x):
        a = x + x
        b = x + x + x
        ctx.mark_non_differentiable(a)
        return a, b

    @staticmethod
    def backward(ctx, grad_a, grad_b):
        grad_x = 3*grad_b
        return grad_x

data = torch.ones([2, 3], dtype=torch.float64, requires_grad=True)
a, b = cus_func.apply(data)
b.sum().backward()

result = data.grad

# torch.autograd.function.FunctionCtx.save_for_backward
# Inherit from Function
class cus_tanh(Function):
    @staticmethod
    def forward(ctx, x):
        # ctx is a context object that store some objects for backward.
        y = torch.tanh(x)
        # Pass tensors to backward.
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, dy):
        grad = dy + 1
        return grad

data = torch.ones([2, 3], dtype=torch.float64, requires_grad=True)
z = cus_tanh.apply(data)
z.sum().backward()

result = data.grad

# torch.autograd.function.FunctionCtx.set_materialize_grads
# Inherit from Function
class cus_tanh(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.set_materialize_grads(False)
        return x+x+x, x+x

    @staticmethod
    def backward(ctx, grad, grad2):
        assert grad2==None
        return grad

x = torch.ones([1], dtype=torch.float64)
x.requires_grad = True
cus_tanh.apply(x)[0].backward()

result = x.grad
result.requires_grad = False

# Inherit from Function
class cus_tanh(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.set_materialize_grads(value=False)
        return x+x+x, x+x

    @staticmethod
    def backward(ctx, grad, grad2):
        assert grad2==None
        return grad

x = torch.ones([1], dtype=torch.float64)
x.requires_grad = True
cus_tanh.apply(x)[0].backward()

result = x.grad
result.requires_grad = False

# torch.bernoulli
a = torch.tensor([0.8, 0.1, 0.4])
result = torch.bernoulli(a)

a = torch.ones(3, 3)
result = torch.bernoulli(a)

# torch.combinations
x = torch.tensor([1, 2, 3], dtype=torch.int32)
result = torch.combinations(input=x)

x = torch.tensor([1, 2, 3], dtype=torch.int32)
result = torch.combinations(x)

# torch.count_nonzero
a = torch.tensor([[0., 1.1, 1.2], [0., 0., 1.3], [0., 0., 0.]])
result = torch.count_nonzero(a)

a = torch.tensor([[0., 1.1, 1.2], [0., 0., 1.3], [0., 0., 0.]])
result = torch.count_nonzero(input=a, dim=1)

# torch.cumulative_trapezoid
result = torch.cumulative_trapezoid(torch.tensor([1.0, 1, 1, 0, 1]))

y = torch.tensor([1, 1, 1, 0, 1]).type(torch.float32)
x = torch.tensor([1, 2, 3, 0, 1]).type(torch.float32)
result = torch.cumulative_trapezoid(y, x)

# torch.frexp
x = torch.tensor([10.0, -2.5, 0.0, 3.14])
result, exponent = torch.frexp(x)

x = torch.tensor([[128.0, 64.0], [-32.0, 16.0]])
result, ex = torch.frexp(x)

# torch.hypot
a = torch.tensor([1., 2, 3])
b = torch.tensor([4., 5, 6])
result = torch.hypot(a, b)

a = torch.tensor([1.])
b = torch.tensor([4., 5, 6])
result = torch.hypot(input=a, other=b)

# torch.isneginf
x = torch.tensor([-float('inf'), float('inf'), 1.2])
result = torch.isneginf(x)

out = torch.tensor([False, False, False])
x = torch.tensor([-float('inf'), float('inf'), 1.2])
result = torch.isneginf(input=x, out = out)

# torch.isposinf
x = torch.tensor([-float('inf'), float('inf'), 1.2])
result = torch.isposinf(x)

out = torch.tensor([False, False, False])
x = torch.tensor([-float('inf'), float('inf'), 1.2])
result = torch.isposinf(input=x, out = out)

# torch.isreal
result = torch.isreal(torch.tensor([1, 1+1j, 2+0j]))

result = torch.isreal(input=torch.tensor([-0., -2.1, 2.5]))

# torch.kron
mat1 = torch.eye(2)
mat2 = torch.ones(2, 2)
result = torch.kron(mat1, mat2)

mat1 = torch.eye(2)
mat2 = torch.ones(2, 2)
result = torch.kron(input=mat1, other=mat2)

# torch.kthvalue
x = torch.tensor([1., 2., 3., 4., 5.])
result = torch.kthvalue(x, 4)

x = torch.tensor([[ 1., 2., 3.], [ 4., 5., 6.]])
result = torch.kthvalue(x, 2, 0, True)

# torch.logcumsumexp
x = torch.tensor([[0.56, 0.34, 0.78], [0.98, 0.21, 1.78]])
result = torch.logcumsumexp(x, 0)

x = torch.tensor([[0.56, 0.34, 0.78], [0.98, 0.21, 1.78]])
result = torch.logcumsumexp(x, 1)

# torch.mode
input = torch.tensor([[[1,2,2],[2,3,3]],[[0,5,5],[9,9,0]]])
result, index = torch.mode(input)

input = torch.tensor([[[1,2,2],[2,3,3]],[[0,5,5],[9,9,0]]])
result = torch.mode(input)
result, index = result[0], result[1]

# torch.mv
a = torch.tensor([[1., 2.], [4., 5.]])
b = torch.tensor([1., 3.])
result = torch.mv(a, b)

a = torch.tensor([[1., 2.], [4., 5.]])
b = torch.tensor([1., 3.])
result = torch.mv(input=a, vec=b)

# torch.nn.Module.named_buffers
class SubModel(nn.Module):
    def __init__(self):
        super(SubModel, self).__init__()
        self.register_buffer('buf1', torch.tensor([1.,2.,4.,5.]))
        self.register_buffer('buf4', torch.tensor([1.,2.,4.,5.]))
        self.register_buffer('buf5', torch.tensor([1.,2.,4.,5.]))

    def forward(self, x):
        return x

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.sub = SubModel()
        self.register_buffer('buf1', torch.tensor([1.,2.,4.,5.]))
        self.register_buffer('buf2', torch.tensor([1.,2.,4.,5.]))
        self.register_buffer('buf3', torch.tensor([1.,2.,4.,5.]))

    def forward(self, x):
        return x

model = Model()
result = []
for name, buf in model.named_buffers():
    result.append((name, buf))

class SubModel(nn.Module):
    def __init__(self):
        super(SubModel, self).__init__()
        self.register_buffer('buf1', torch.tensor([1.,2.,4.,5.]))
        self.register_buffer('buf4', torch.tensor([1.,2.,4.,5.]))
        self.register_buffer('buf5', torch.tensor([1.,2.,4.,5.]))

    def forward(self, x):
        return x

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.sub = SubModel()
        self.register_buffer('buf1', torch.tensor([1.,2.,4.,5.]))
        self.register_buffer('buf2', torch.tensor([1.,2.,4.,5.]))
        self.register_buffer('buf3', torch.tensor([1.,2.,4.,5.]))

    def forward(self, x):
        return x

model = Model()
result = []
for name, buf in model.named_buffers(prefix='wfs'):
    result.append((name, buf))

# torch.nn.ParameterDict
choices = nn.ParameterDict({
    f"param_{i}": nn.Parameter(torch.ones(i + 1, i + 1)) for i in range(10)
})
result = list(choices)

choices = nn.ParameterDict()
result = list(choices)

# torch.nn.attention.SDPBackend
modified_backend_state = [torch.nn.attention.SDPBackend.MATH]

np.random.seed(100)
x_data = np.random.randn(2, 2, 2, 2)
x = torch.tensor(x_data, dtype=torch.float32)

with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
    result = torch.nn.functional.scaled_dot_product_attention(x, x, x).to(torch.float32)
    current_backends = torch.nn.attention._cur_sdpa_kernel_backends()
    assert current_backends == modified_backend_state

original_backend_state = set(torch.nn.attention._cur_sdpa_kernel_backends())
modified_backend_state = [torch.nn.attention.SDPBackend.MATH]

np.random.seed(100)
x_data = np.random.randn(2, 2, 2, 2)
x = torch.tensor(x_data, dtype=torch.float32)

# Check original state
current_backends = set(torch.nn.attention._cur_sdpa_kernel_backends())
assert current_backends == original_backend_state, f"Expected {original_backend_state}, got {current_backends}"

with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
    output1 = torch.nn.functional.scaled_dot_product_attention(x, x, x).to(torch.float32)
    current_backends = torch.nn.attention._cur_sdpa_kernel_backends()
    assert current_backends == modified_backend_state, f"Expected {modified_backend_state}, got {current_backends}"

    output2 = torch.nn.functional.scaled_dot_product_attention(x, x, x).to(torch.float32)
    current_backends = torch.nn.attention._cur_sdpa_kernel_backends()
    assert current_backends == modified_backend_state, f"Expected {modified_backend_state}, got {current_backends}"

    output3 = torch.nn.functional.scaled_dot_product_attention(x, x, x).to(torch.float32)
    current_backends = torch.nn.attention._cur_sdpa_kernel_backends()
    assert current_backends == modified_backend_state, f"Expected {modified_backend_state}, got {current_backends}"

# Check back to original state
current_backends = set(torch.nn.attention._cur_sdpa_kernel_backends())
assert current_backends == original_backend_state, f"Expected {original_backend_state}, got {current_backends}"

result = output1 + output2 + output3

# torch.nn.attention.SDPBackend.FLASH_ATTENTION
modified_backend_state = {
    torch.nn.attention.SDPBackend.MATH,
    torch.nn.attention.SDPBackend.FLASH_ATTENTION,
}

np.random.seed(100)
x_data = np.random.randn(2, 2)
x = torch.tensor(x_data, dtype=torch.float32)

with torch.nn.attention.sdpa_kernel([
    torch.nn.attention.SDPBackend.MATH,
    torch.nn.attention.SDPBackend.FLASH_ATTENTION,
]):
    # FLASH_ATTENTION may not be supported, but we're not actually doing any sdpa
    x = x + 1
    current_backends = set(torch.nn.attention._cur_sdpa_kernel_backends())
    assert current_backends == modified_backend_state, f"Expected {modified_backend_state}, got {current_backends}"
    x = x + 1

result = x

# torch.nn.attention.SDPBackend.MATH
modified_backend_state = [torch.nn.attention.SDPBackend.MATH]

np.random.seed(100)
x_data = np.random.randn(2, 2, 2, 2)
x = torch.tensor(x_data, dtype=torch.float32)

with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
    result = torch.nn.functional.scaled_dot_product_attention(x, x, x).to(torch.float32)
    current_backends = torch.nn.attention._cur_sdpa_kernel_backends()
    assert current_backends == modified_backend_state

original_backend_state = set(torch.nn.attention._cur_sdpa_kernel_backends())
modified_backend_state = [torch.nn.attention.SDPBackend.MATH]

np.random.seed(100)
x_data = np.random.randn(2, 2, 2, 2)
x = torch.tensor(x_data, dtype=torch.float32)

# Check original state
current_backends = set(torch.nn.attention._cur_sdpa_kernel_backends())
assert current_backends == original_backend_state, f"Expected {original_backend_state}, got {current_backends}"

with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
    output1 = torch.nn.functional.scaled_dot_product_attention(x, x, x).to(torch.float32)
    current_backends = torch.nn.attention._cur_sdpa_kernel_backends()
    assert current_backends == modified_backend_state, f"Expected {modified_backend_state}, got {current_backends}"

    output2 = torch.nn.functional.scaled_dot_product_attention(x, x, x).to(torch.float32)
    current_backends = torch.nn.attention._cur_sdpa_kernel_backends()
    assert current_backends == modified_backend_state, f"Expected {modified_backend_state}, got {current_backends}"

    output3 = torch.nn.functional.scaled_dot_product_attention(x, x, x).to(torch.float32)
    current_backends = torch.nn.attention._cur_sdpa_kernel_backends()
    assert current_backends == modified_backend_state, f"Expected {modified_backend_state}, got {current_backends}"

# Check back to original state
current_backends = set(torch.nn.attention._cur_sdpa_kernel_backends())
assert current_backends == original_backend_state, f"Expected {original_backend_state}, got {current_backends}"

result = output1 + output2 + output3

# torch.nn.attention._cur_sdpa_kernel_backends
modified_backend_state = [torch.nn.attention.SDPBackend.MATH]

np.random.seed(100)
x_data = np.random.randn(2, 2, 2, 2)
x = torch.tensor(x_data, dtype=torch.float32)

with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
    result = torch.nn.functional.scaled_dot_product_attention(x, x, x).to(torch.float32)
    current_backends = torch.nn.attention._cur_sdpa_kernel_backends()
    assert current_backends == modified_backend_state

original_backend_state = set(torch.nn.attention._cur_sdpa_kernel_backends())
modified_backend_state = [torch.nn.attention.SDPBackend.MATH]

np.random.seed(100)
x_data = np.random.randn(2, 2, 2, 2)
x = torch.tensor(x_data, dtype=torch.float32)

# Check original state
current_backends = set(torch.nn.attention._cur_sdpa_kernel_backends())
assert current_backends == original_backend_state, f"Expected {original_backend_state}, got {current_backends}"

with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
    output1 = torch.nn.functional.scaled_dot_product_attention(x, x, x).to(torch.float32)
    current_backends = torch.nn.attention._cur_sdpa_kernel_backends()
    assert current_backends == modified_backend_state, f"Expected {modified_backend_state}, got {current_backends}"

    output2 = torch.nn.functional.scaled_dot_product_attention(x, x, x).to(torch.float32)
    current_backends = torch.nn.attention._cur_sdpa_kernel_backends()
    assert current_backends == modified_backend_state, f"Expected {modified_backend_state}, got {current_backends}"

    output3 = torch.nn.functional.scaled_dot_product_attention(x, x, x).to(torch.float32)
    current_backends = torch.nn.attention._cur_sdpa_kernel_backends()
    assert current_backends == modified_backend_state, f"Expected {modified_backend_state}, got {current_backends}"

# Check back to original state
current_backends = set(torch.nn.attention._cur_sdpa_kernel_backends())
assert current_backends == original_backend_state, f"Expected {original_backend_state}, got {current_backends}"

result = output1 + output2 + output3

# torch.nn.functional.pixel_shuffle
input = torch.randn(1, 9, 4, 4)
result = F.pixel_shuffle(input, 3)

input = torch.randn(1, 9, 4, 4)
result = F.pixel_shuffle(input, upscale_factor=3)

# torch.nn.functional.pixel_unshuffle
input = torch.randn(1, 1, 12, 12)
result = F.pixel_unshuffle(input, 3)

input = torch.randn(1, 1, 12, 12)
result = F.pixel_unshuffle(input, downscale_factor=3)

# torch.poisson
rates = torch.rand(4, 4) * 5
result = torch.poisson(rates)

rates = torch.tensor([[1., 3., 4.], [2., 3., 6.]])
result = torch.poisson(rates)

# torch.trapezoid
y = torch.tensor([1.0, 1, 1, 0, 1])
result = torch.trapezoid(y)

y = torch.tensor([1, 1, 1, 0, 1]).type(torch.float32)
x = torch.tensor([1, 2, 3, 0, 1]).type(torch.float32)
result = torch.trapezoid(y=y, x=x)

# torch.utils.data.dataset.ChainDataset
class MyIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, start, end):
        super(MyIterableDataset).__init__()
        assert end > start, "this example code only works with end >= start"
        self.start = start
        self.end = end

    def __iter__(self):
        iter_start = self.start
        iter_end = self.end
        return iter(range(iter_start, iter_end))


dataset = torch.utils.data.ChainDataset([MyIterableDataset(start=3, end=7), MyIterableDataset(start=3, end=7)])
result = []
for d in dataset:
    result.append(d)

class MyIterableDataset(torch_data.IterableDataset):
    def __init__(self, start, end):
        super(MyIterableDataset).__init__()
        assert end > start, "this example code only works with end >= start"
        self.start = start
        self.end = end

    def __iter__(self):
        iter_start = self.start
        iter_end = self.end
        return iter(range(iter_start, iter_end))


dataset = torch_data.ChainDataset([MyIterableDataset(start=1, end=10), MyIterableDataset(start=1, end=3)])
result = []
for d in dataset:
    result.append(d)

# torch.utils.data.dataset.IterableDataset
class MyIterableDataset(IterableDataset):
    def __init__(self, start, end):
        super(MyIterableDataset).__init__()
        assert end > start, "this example code only works with end >= start"
        self.start = start
        self.end = end

    def __iter__(self):
        return iter(range(self.start, self.end))

ds = MyIterableDataset(start=3, end=7)
result = []
for i in ds:
    result.append(i)

class MyIterableDataset(torch_data.IterableDataset):
    def __init__(self, start, end):
        super(MyIterableDataset).__init__()
        assert end > start, "this example code only works with end >= start"
        self.start = start
        self.end = end

    def __iter__(self):
        return iter(range(self.start, self.end))

ds = MyIterableDataset(start=3, end=7)
result = next(ds.__iter__())

# torch.utils.data.default_collate
result = torch.tensor(default_collate([0, 1, 2, 3]))

result = default_collate(['a', 'b', 'c'])

# torch.Tensor.kron
x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
result = x.kron(y)

x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
result = torch.Tensor.kron(x, y)

# torch.Tensor.log_normal_
x = torch.empty(2, 3)
result = x.log_normal_()

x = torch.empty(2, 3)
result = x.log_normal_(1.0, 0.5)

# torch.distributed.ReduceOp
op_type = torch.distributed.ReduceOp

op_type = ReduceOp

# torch.distributed.batch_isend_irecv
ops = []
result = torch.distributed.batch_isend_irecv(ops)

ops = []
result = torch.distributed.batch_isend_irecv(ops, op=torch.distributed.ReduceOp.MIN)

# torch.distributed.get_backend
result = torch.distributed.get_backend()

result = torch.distributed.get_backend(group=None)

# torch.distributed.get_rank
result = torch.distributed.get_rank()

result = torch.distributed.get_rank(group=None)

# torch.distributed.get_world_size
result = torch.distributed.get_world_size()

result = torch.distributed.get_world_size(group=None)

# torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION
mode = torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION

mode = SDPBackend.EFFICIENT_ATTENTION

# torch.nn.attention.SDPBackend.ERROR
mode = torch.nn.attention.SDPBackend.ERROR

mode = SDPBackend.ERROR

# transformers.PreTrainedTokenizer
cls = transformers.PreTrainedTokenizer

tokenizer = transformers.PreTrainedTokenizer()

# torch.distributed.ReduceOp.MAX
value = torch.distributed.ReduceOp.MAX

value = ReduceOp.MAX

# torch.distributed.ReduceOp.MIN
value = torch.distributed.ReduceOp.MIN

value = ReduceOp.MIN

# torch.distributed.ReduceOp.SUM
value = torch.distributed.ReduceOp.SUM

value = ReduceOp.SUM

# torch.nn
module = torch.nn

module = nn
