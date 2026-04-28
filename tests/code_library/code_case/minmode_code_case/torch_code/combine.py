import torch
import torch.nn as nn
from torch.autograd import Function
from collections import OrderedDict
import numpy as np
import torch.nn.functional as F
print('#########################case1#########################')
input = torch.tensor([[[[1.1524, 0.4714, 0.2857], [-1.2533, -0.9829, -
    1.0981], [0.1507, -1.1431, -2.0361]], [[0.1024, -0.4482, 0.4137], [
    0.9385, 0.4565, 0.7702], [0.4135, -0.2587, 0.0482]]]])
m = torch.nn.Upsample(scale_factor=2, mode='bilinear',
    recompute_scale_factor=True)
result = m(input)
print('#########################case2#########################')
x = torch.tensor([[[[1.0, 2.0, 3.0], [2.0, 3.0, 4]]], [[[1.0, 2.0, 3.0], [
    2.0, 3.0, 4]]]])
result = F.interpolate(x, None, [2, 1], 'bilinear', True, True, True)
print('#########################case3#########################')
a = torch.tensor([[1.0, 3.0, 8.0, 11.0, 56.0], [15.0, 30.0, 7.0, 14.0, 90.0
    ], [10.0, 313.0, 78.0, 110.0, 34.0], [33.0, 23.0, 18.0, 9.0, 41.0]])
result = a.tril(diagonal=-1)
print('#########################case4#########################')
input = torch.tensor([[[1.1524, 0.4714, 0.2857], [-1.2533, -0.9829, -1.0981
    ], [0.1507, -1.1431, -2.0361]], [[0.1024, -0.4482, 0.4137], [0.9385, 
    0.4565, 0.7702], [0.4135, -0.2587, 0.0482]]])
data = torch.tensor([1.0, 1.0, 1.0])
result = torch.layer_norm(input=input, normalized_shape=[3], weight=data,
    bias=data)
print('#########################case5#########################')
src = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
index = torch.tensor([[0, 1, 1], [0, 0, 1]])
input = torch.tensor([[10.0, 30.0, 20.0], [60.0, 40.0, 50.0]])
re_type = 'amax'
result = input.scatter_reduce(dim=0, index=index, src=src, reduce=re_type)
print('#########################case6#########################')
out = torch.zeros(3, 5, dtype=torch.bool)
x = torch.tensor([[-float('inf'), float('inf'), 1.2, 0.0, 2.5], [-1.35, -
    float('inf'), 0.18, -0.33, float('inf')], [-float('inf'), float('inf'),
    1.0, 2.0, 4.0]])
result = torch.isposinf(out=out, input=x)
print('#########################case7#########################')
input = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
result = input.baddbmm(batch1=torch.tensor([[[4.0, 5.0, 6.0], [1.0, 2.0, 
    3.0]]]), batch2=torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 
    8.0, 9.0]]]), beta=3, alpha=3)
print('#########################case8#########################')
input = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
result = input.baddbmm_(batch1=torch.tensor([[[4.0, 5.0, 6.0], [1.0, 2.0, 
    3.0]]]), batch2=torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 
    8.0, 9.0]]]), beta=3, alpha=3)
print('#########################case9#########################')
input = torch.tensor([[[0.9041, 0.0196], [-0.3108, -2.4423]], [[0.5012, -
    0.1234], [0.7891, 0.3456]]])
other = torch.tensor([[[0.2341, 0.2539], [-0.6256, -0.6448]], [[0.1122, -
    0.3344], [0.5566, 0.7788]]])
result = input.atan2(other)
print('#########################case10#########################')
src = torch.tensor([0.0 + 3.5j, -1 + 4.2j, 2.34 - 5.2j, -3.45 + 7.9j, -0.34 -
    8.2j, 0.23 + 9.2j, 1.0 + 1.0j, 2.0 + 0.5j, 3.0 - 1.0j], dtype=torch.
    complex128)
result = src.bfloat16().float()
print('#########################case11#########################')
x = torch.tensor([[1, 3, 9, 7, 5], [2, 4, 6, 8, 10]])
values = torch.tensor([[3, 6, 9], [3, 6, 9]])
out = torch.tensor([[3, 6, 9], [3, 6, 9]])
sorter = torch.argsort(x)
result = torch.searchsorted(sorted_sequence=x, input=values, out_int32=
    False, right=True, side='right', sorter=sorter, out=out)
print('#########################case12#########################')
x = torch.tensor([[[[[-0.8658, 1.0869, -2.1977], [-2.1073, 1.0974, -1.4485],
    [0.588, -0.7189, 0.1089]], [[1.3036, 0.3086, -1.2245], [-0.6707, -
    0.0195, -0.1474], [0.2727, -0.4938, -0.6854]], [[0.5525, 1.0111, -
    0.1847], [0.1111, -0.6373, -0.222], [-0.5963, 0.7734, 0.0409]]]]])
model = nn.MaxPool3d(kernel_size=2, stride=1, padding=1, dilation=2,
    return_indices=True, ceil_mode=True)
result, indices = model(x)
print('#########################case13#########################')
t = torch.tensor([[3.0 + 3.0j, 2.0 + 2.0j, 3.0 + 3.0j], [2.0 + 2.0j, 2.0 + 
    2.0j, 3.0 + 3.0j]])
out = torch.randn(2, 1)
result = torch.fft.irfft(input=t, n=1, dim=1, norm='forward', out=out)
print('#########################case14#########################')


class RandomDataset(torch.utils.data.Dataset):

    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __getitem__(self, idx):
        image = np.arange(5).astype('float32')
        label = np.array([idx]).astype('int64')
        return torch.tensor(image), torch.tensor(label)

    def __len__(self):
        return self.num_samples


dataset = torch.utils.data.ConcatDataset(datasets=[RandomDataset(2),
    RandomDataset(2)])
result = []
for i in range(len(dataset)):
    result.append(dataset[i])
print('#########################case15#########################')


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
output.backward(torch.tensor([1.0, 1.0, 1.0]))
result = data.grad
result.requires_grad = False
print('#########################case16#########################')
embedding_matrix = torch.Tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 
    2.0, 2.0], [3.0, 3.0, 3.0]])
x = torch.tensor(np.array([[0, 1], [2, 3]]))
result = torch.nn.functional.embedding(input=x, weight=embedding_matrix,
    padding_idx=0, max_norm=2, norm_type=2.0, scale_grad_by_freq=False,
    sparse=True)
print('#########################case17#########################')
x1 = torch.tensor([[1.683, 0.0526], [-0.0696, 0.6366], [-1.0091, 1.3363]])
x2 = torch.tensor([[-0.0629, 0.2414], [-0.9701, -0.4455]])
result = torch.cdist(x1=x1, x2=x2, p=1.0, compute_mode=
    'use_mm_for_euclid_dist_if_necessary')
print('#########################case18#########################')
x = torch.tensor([1.0, 2.0, 3.0])
module1 = torch.nn.Module()
module1.register_buffer('buffer', x)
module2 = torch.nn.Module()
module2.add_module(name='submodule', module=module1)
result = module2.submodule.buffer
print('#########################case19#########################')
t = torch.tensor([[3.0 + 3.0j, 2.0 + 2.0j, 3.0 + 3.0j], [2.0 + 2.0j, 2.0 + 
    2.0j, 3.0 + 3.0j]])
out = torch.randn(2, 5)
result = torch.fft.hfft2(out=out, norm='backward', dim=(0, 1), s=(2, 5),
    input=t)
print('#########################case20#########################')
t = torch.tensor([[3.0 + 3.0j, 2.0 + 2.0j, 3.0 + 3.0j], [2.0 + 2.0j, 2.0 + 
    2.0j, 3.0 + 3.0j]])
out = torch.randn(2, 3)
result = torch.fft.irfft2(out=out, norm='forward', dim=(0, 1), s=(2, 3),
    input=t)
print('#########################case21#########################')
input = torch.tensor([[1.3398, 0.2663, -0.2686, 0.245], [-0.7401, -0.8805, 
    -0.3402, -1.1936], [0.4907, -1.3948, -1.0691, -0.3132], [-1.6092, 
    0.5419, -0.2993, 0.3195]])
result = torch.argmax(input, dim=1, keepdim=True)
print('#########################case22#########################')
input = torch.tensor([[1.3398, 0.2663, -0.2686, 0.245], [-0.7401, -0.8805, 
    -0.3402, -1.1936], [0.4907, -1.3948, -1.0691, -0.3132], [-1.6092, 
    0.5419, -0.2993, 0.3195]])
result = torch.argmin(input, dim=1, keepdim=True)
print('#########################case23#########################')
x = torch.randn(5, 16, 50)
model = nn.Conv1d(16, 33, 5, stride=1, padding=4, dilation=3, groups=1,
    bias=True, padding_mode='zeros', device='cpu', dtype=torch.float32)
result = model(x)
print('#########################case24#########################')
input = torch.tensor([[1.3398, 0.2663, -0.2686, 0.245], [-0.7401, -0.8805, 
    -0.3402, -1.1936], [0.4907, -1.3948, -1.0691, -0.3132], [-1.6092, 
    0.5419, -0.2993, 0.3195]])
dim = 1
result = input.argsort(dim=dim, descending=False)
print('#########################case25#########################')
x = torch.tensor([1.0, 2.0, 3.0])
module1 = torch.nn.Module()
module1.register_buffer('buffer', x)
module2 = torch.nn.Module()
module2.register_module(name='submodule', module=module1)
result = module2.submodule.buffer
print('#########################case26#########################')
x = torch.tensor([[[[[-0.6, 0.8, -0.5], [-0.5, 0.2, 1.2], [1.4, 0.3, -0.2]]]]])
grid = torch.tensor([[[[[0.2, 0.2, 0.3], [-0.4, 0.2, -0.3], [-0.9, 0.2, 0.3
    ], [-0.9, 0.9, -0.6]], [[0.2, 0.2, 0.3], [-0.4, 0.2, -0.3], [-0.9, 0.2,
    0.3], [-0.9, 0.9, -0.6]]]]])
result = F.grid_sample(input=x, grid=grid, mode='bilinear', padding_mode=
    'border', align_corners=True)
print('#########################case27#########################')
l = nn.Linear(2, 2)
net = nn.Sequential(OrderedDict([('wfs', l), ('wfs1', l)]))
memo = set()
z = net.named_modules(memo=memo, prefix='wfs', remove_duplicate=False)
name_list = []
for idx, m in enumerate(z):
    name_list.append(m[0])
result = name_list
print('#########################case28#########################')
x = torch.randn(5, 16, 50, 20, 20)
model = nn.Conv3d(16, 33, (3, 3, 5), (2, 2, 1), (4, 2, 2), (3, 1, 1), 1,
    bias=False, padding_mode='zeros', device='cpu', dtype=torch.float32)
result = model(x)
print('#########################case29#########################')
pool = nn.MaxPool2d(2, stride=2, return_indices=True)
unpool = nn.MaxUnpool2d(2)
input = torch.tensor([[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 
    10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]]])
output, indices = pool(input)
result = unpool(output, indices)
print('#########################case30#########################')
x = torch.tensor([[[[[-1.1494, -1.3829], [0.4995, -1.3094]], [[1.0015, 
    1.4919], [-1.5187, 0.0235]]]]])
model = nn.AdaptiveAvgPool3d(1)
result = model(x)
print('#########################case31#########################')
x = torch.tensor([[[[[-1.1494, -1.3829], [0.4995, -1.3094]], [[1.0015, 
    1.4919], [-1.5187, 0.0235]]]]])
model = nn.AdaptiveMaxPool3d(1)
result = model(x)
print('#########################case32#########################')
x = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
x.renorm_(1, 0, 5)
print('#########################case33#########################')
x = torch.tensor([[4.0, 5.0, 6.0], [1.0, 2.0, 3.0], [4.0, 9.0, 10.0]])
y = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
out = torch.tensor([[4.0, 5.0, 6.0], [1.0, 2.0, 3.0], [4.0, 9.0, 10.0]])
result = torch.linalg.matmul(x, y, out=out)
print('#########################case34#########################')


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
print('#########################case35#########################')
x = torch.tensor([[[[[-1.1494, -1.3829], [0.4995, -1.3094]], [[1.0015, 
    1.4919], [-1.5187, 0.0235]]]]])
result = nn.functional.adaptive_max_pool3d(x, 1, False)
print('#########################case36#########################')
x = torch.tensor([[4.0, 5.0, 6.0], [1.0, 2.0, 3.0], [4.0, 9.0, 10.0]])
y = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
out = torch.tensor([[4.0, 5.0, 6.0], [1.0, 2.0, 3.0], [4.0, 9.0, 10.0]])
result = torch.matmul(input=x, other=y, out=out)
print('#########################case37#########################')
x = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
out = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
result = torch.cumsum(x, 0, out=out)
print('#########################case38#########################')
x = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
out = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
result = torch.cumprod(x, 0, out=out)
print('#########################case39#########################')
out = torch.zeros(3, 5, dtype=torch.bool)
x = torch.tensor([[-float('inf'), float('inf'), 1.2, 0.0, 2.5], [-1.35, -
    float('inf'), 0.18, -0.33, float('inf')], [-float('inf'), float('inf'),
    1.0, 2.0, 4.0]])
result = torch.isneginf(out=out, input=x)
print('#########################case40#########################')
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
result = torch.tanh(x)
print('#########################case41#########################')
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
result = torch.sigmoid(x)
print('#########################case42#########################')
x = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
result = x.renorm(1, 0, 5)
print('#########################case43#########################')
result = torch.quantile(torch.tensor([[0.0795, -1.2117, 0.9765], [1.1707, 
    0.6706, 0.4884]], dtype=torch.float64), 0.6, dim=1, keepdim=True)
print('#########################case44#########################')
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
result = nn.Tanh()(x)
print('#########################case45#########################')
x = torch.tensor([[[[-0.0878, 0.3378, 0.0547, 1.2068], [0.4212, -1.6113, 
    0.7277, 0.0766], [0.8189, 0.0958, 1.778, 1.1192], [0.7286, -0.1988, 
    1.0519, 0.9217]], [[0.0088, -1.9815, -0.3543, 0.1712], [-0.183, 0.0325,
    -0.1784, 0.1072], [1.1752, -0.0234, -1.0873, -0.5568], [0.4471, 0.4073,
    -1.6031, -0.031]]]])
weight = torch.tensor([1.3, 1.2])
bias = torch.tensor([0.1, 0.2])
result = F.group_norm(x, 2, weight, bias, 1e-05)
print('#########################case46#########################')
t = torch.tensor([[3.0 + 3.0j, 2.0 + 2.0j, 3.0 + 3.0j], [2.0 + 2.0j, 2.0 + 
    2.0j, 3.0 + 3.0j]])
out = torch.complex(torch.randn(2, 3), torch.randn(2, 3))
result = torch.fft.fftn(t, s=(2, 3), dim=(0, 1), norm='forward', out=out)
print('#########################case47#########################')
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
m = torch.nn.GLU()
result = m(x)
print('#########################case48#########################')
t = torch.tensor([[3.0 + 3.0j, 2.0 + 2.0j, 3.0 + 3.0j], [2.0 + 2.0j, 2.0 + 
    2.0j, 3.0 + 3.0j]])
out = torch.complex(torch.randn(2, 3), torch.randn(2, 3))
result = torch.fft.ifft2(t, s=(2, 3), dim=(0, 1), norm='forward', out=out)
print('#########################case49#########################')
t = torch.tensor([[3.0 + 3.0j, 2.0 + 2.0j, 3.0 + 3.0j], [2.0 + 2.0j, 2.0 + 
    2.0j, 3.0 + 3.0j]])
out = torch.complex(torch.randn(2, 3), torch.randn(2, 3))
result = torch.fft.ifftn(t, s=(2, 3), dim=(0, 1), norm='forward', out=out)
print('#########################case50#########################')


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
result = data.grad
print('#########################case51#########################')
t = torch.tensor([[3.0 + 3.0j, 2.0 + 2.0j, 3.0 + 3.0j], [2.0 + 2.0j, 2.0 + 
    2.0j, 3.0 + 3.0j]])
real = torch.randn(3, 2)
imag = torch.randn(3, 2)
out = torch.view_as_complex(torch.stack((real, imag), dim=-1))
result = torch.fft.fft2(input=t, s=None, dim=(-2, -1), norm='forward', out=out)
print('#########################case52#########################')
x = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
result = torch.renorm(x, 1, 0, 5)
print('#########################case53#########################')
a = torch.tensor([[0.218, 1.0558, 0.1608, 0.9245], [1.3794, 1.409, 0.2514, 
    -0.8818], [-0.4561, 0.5123, 1.7505, -0.4094]])
result = a.unflatten(1, (2, 2))
print('#########################case54#########################')
input = torch.tensor([[[1.1524, 0.4714, 0.2857, 0.4586, 0.9876], [-1.2533, 
    -0.9829, -1.0981, 0.7655, 0.8541], [0.1507, -1.1431, -2.0361, 0.2344, 
    0.5675]]])
result, indices = F.max_pool1d(input, 5, stride=2, padding=2, ceil_mode=
    True, return_indices=True)
print('#########################case55#########################')
a = 3
out = torch.tensor([2.0, 3.0], dtype=torch.float64)
result = torch.ones(a, a, out=out, dtype=torch.float64, device=torch.device
    ('cpu'), requires_grad=True, pin_memory=False)
print('#########################case56#########################')
a = 3
out = torch.tensor([2.0, 3.0], dtype=torch.float64)
result = torch.empty(a, a, out=out, dtype=torch.float64, device=torch.
    device('cpu'), requires_grad=True, pin_memory=False)
print('#########################case57#########################')
torch.manual_seed(100)
result = torch.initial_seed()
print('#########################case58#########################')
x = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
y = torch.tensor([[8.0, 3.0, 3.0], [2.0, 3.0, 4.0]])
model = nn.CosineSimilarity()
result = model(x, y)
print('#########################case59#########################')
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
result = nn.Sigmoid()(x)
print('#########################case60#########################')
result = torch.nonzero(torch.tensor([[0.6, 0.0, 0.0, 0.0], [0.0, 0.4, 0.0, 
    0.0], [0.0, 0.0, 1.2, 0.0], [0.0, 0.0, 0.0, -0.4]]))
print('#########################case61#########################')
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
result = nn.Softsign()(x)
print('#########################case62#########################')
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
F.leaky_relu_(x)
print('#########################case63#########################')
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
result = nn.LogSigmoid()(x)
print('#########################case64#########################')
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
result = F.glu(x)
print('#########################case65#########################')
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
result = F.gelu(x)
print('#########################case66#########################')
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
result = F.mish(x)
print('#########################case67#########################')
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
result = F.relu(x)
print('#########################case68#########################')
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
result = F.silu(x)
print('#########################case69#########################')
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
result = F.tanh(x)
print('#########################case70#########################')
a = torch.tensor([[0.218, 1.0558, 0.1608, 0.9245], [1.3794, 1.409, 0.2514, 
    -0.8818], [-0.4561, 0.5123, 1.7505, -0.4094]])
result = torch.unflatten(a, 1, (2, 2))
print('#########################case71#########################')
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
result = F.relu_(x)
print('#########################case72#########################')
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
result = F.dropout(x)
print('#########################case73#########################')
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
result = F.sigmoid(x)
print('#########################case74#########################')
x = torch.tensor([[[-1.302, -0.1005, 0.5766, 0.6351, -0.8893, 0.0253, -
    0.1756, 1.2913], [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.124, 
    -1.187, -1.8816]]])
result = F.softplus(x)
print('#########################case75#########################')
out = torch.empty([6], dtype=torch.int32)
result = torch.randperm(6, out=out, dtype=torch.int32, layout=torch.strided,
    device=torch.device('cpu'), pin_memory=False, requires_grad=False)
print('#########################case76#########################')
torch.cuda.manual_seed(123)
result = torch.cuda.initial_seed()
print('#########################case77#########################')
x = torch.tensor([1.0, 2.0, 3.0])
result = x.new_empty(size=(2, 3), dtype=torch.float64, device='cpu',
    requires_grad=True, layout=torch.strided, pin_memory=False)
print('#########################case78#########################')
x = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]],
    requires_grad=True)
linear = torch.nn.Linear(3, 4, bias=False)
linear.weight.data.fill_(0.1)
y = linear(x)
y.detach_()
print('#########################case79#########################')
x = torch.zeros(5, 16, 50, 100).to(torch.float64)
model = nn.Conv2d(16, 33, (3, 5), (2, 1), (4, 2), (3, 1), 1, bias=False,
    padding_mode='zeros', device='cpu', dtype=torch.float64)
result = model(x) * 0
print('#########################case80#########################')
x = torch.tensor([[-0.4264, 0.0255, -0.1064], [0.8795, -0.2429, 0.1374], [
    0.1029, -0.6482, -1.63]])
out = torch.tensor([-0.4264, 0.0255, -0.1064])
result = torch.diag(input=x, diagonal=0, out=out)
print('#########################case81#########################')
input = torch.tensor([[-12.0, -11.0, -10.0, -9.0], [-8.0, -7.0, -6.0, -5.0],
    [-4.0, -3.0, -2.0, -1.0]])
result = torch.functional.norm(input, p=2, dim=1, dtype=torch.float64)
print('#########################case82#########################')
x = torch.tensor([[[[0.9785, 1.2013, 2.4873, -1.1891], [-0.0832, -0.5456, -
    0.5009, 1.5103], [-1.286, 1.0287, -1.3902, 0.4627], [-0.0502, -1.3924, 
    -0.3327, 0.1678]]]])
result = nn.functional.adaptive_max_pool2d(x, 5)
print('#########################case83#########################')
x = torch.tensor([[[[0.9785, 1.2013, 2.4873, -1.1891], [-0.0832, -0.5456, -
    0.5009, 1.5103], [-1.286, 1.0287, -1.3902, 0.4627], [-0.0502, -1.3924, 
    -0.3327, 0.1678]]]])
model = nn.AdaptiveAvgPool2d(5)
result = model(x)
print('#########################case84#########################')
x = torch.tensor([[[[0.9785, 1.2013, 2.4873, -1.1891], [-0.0832, -0.5456, -
    0.5009, 1.5103], [-1.286, 1.0287, -1.3902, 0.4627], [-0.0502, -1.3924, 
    -0.3327, 0.1678]]]])
model = nn.AdaptiveMaxPool2d(5)
result = model(x)
print('#########################case85#########################')
x = torch.tensor([[0.0335, 0.183, -0.1269], [0.1897, -0.1422, -0.494], [-
    0.7674, -0.0134, -0.3733]])
size = 2, 2
stride = 1, 2
results = x.as_strided(size=(2, 2), stride=(2, 2), storage_offset=0)
print('#########################case86#########################')
torch.cuda.manual_seed_all(123)
result = torch.cuda.initial_seed()
print('#########################case87#########################')
result = torch.as_tensor([1, 2, 3])
print('#########################case88#########################')
x = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
y = torch.tensor([[8.0, 3.0, 3.0], [1.4, 3.6, 0.8]])
model = nn.PairwiseDistance()
result = model(x, y)
print('#########################case89#########################')
a = 3
out = torch.tensor([2.0, 3.0], dtype=torch.float64)
result = torch.rand(size=(a, a), generator=None, out=out, dtype=torch.
    float64, device=torch.device('cpu'), requires_grad=True, pin_memory=False)
print('#########################case90#########################')
x = torch.tensor([[-1.0813, -0.8619, 0.7105], [0.0935, 0.138, 2.2112], [-
    0.3409, -0.9828, 0.0289]])
out = torch.tensor([2.0])
result = torch.tril(input=x, diagonal=1, out=out)
print('#########################case91#########################')
x = torch.tensor([[-1.0813, -0.8619, 0.7105], [0.0935, 0.138, 2.2112], [-
    0.3409, -0.9828, 0.0289]])
out = torch.tensor([2.0])
result = torch.triu(input=x, diagonal=1, out=out)
print('#########################case92#########################')
a = 3
out = torch.tensor([2.0, 3.0], dtype=torch.float64)
result = torch.randn(size=(a, a), generator=None, out=out, dtype=torch.
    float64, device=torch.device('cpu'), requires_grad=True, pin_memory=False)
print('#########################case93#########################')
a = 3
out = torch.tensor([2.0, 3.0], dtype=torch.float64)
result = torch.zeros(size=[a, a], out=out, dtype=torch.float64, device=
    torch.device('cpu'), requires_grad=True, pin_memory=False)
print('#########################case94#########################')
x = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
y = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
result = x.cross(other=y, dim=1)
print('#########################case95#########################')
result = torch.full_like(torch.empty(2, 3), 6, dtype=torch.float64, device=
    'cpu', requires_grad=True, memory_format=torch.preserve_format)
print('#########################case96#########################')
result = torch.ones_like(torch.empty(2, 3), dtype=torch.float64,
    requires_grad=True, device=None)
print('#########################case97#########################')
result = torch.empty_like(torch.empty(2, 3), dtype=torch.float64, device=
    None, requires_grad=True)
print('#########################case98#########################')
result = torch.zeros_like(input=torch.empty(2, 3), dtype=torch.float64,
    device=torch.device('cpu'), requires_grad=True)
print('#########################case99#########################')
a = torch.tensor([[[[2.0, 3.0], [3.0, 5.0]], [[5.0, 3.0], [9.0, 5.0]]]])
m = torch.nn.GroupNorm(num_groups=2, num_channels=2, eps=1e-05, affine=True,
    device='cpu', dtype=torch.float32)
result = m(a)
print('#########################case100#########################')
input = torch.tensor([[-12.0, -11.0, -10.0, -9.0], [-8.0, -7.0, -6.0, -5.0],
    [-4.0, -3.0, -2.0, -1.0]])
out = torch.tensor([1.0], dtype=torch.float64)
result = torch.norm(input, 2, 1, True, out, torch.float64)
