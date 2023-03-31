import torch
# torch.Tensor.index_copy_
## case 1: index axis = 0
x = torch.zeros(5, 3)
t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
index = torch.tensor([0, 4, 2])
x.index_copy_(0, index, t)
print('[torch.Tensor.index_copy_ case-1]: ', x)

## Case2:  index axis !=0  high dimension data that x and tensor must have same shape
x = torch.zeros(2, 1, 3, 3)
t = torch.tensor([
    [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]],
    [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]], dtype=torch.float)
index = torch.tensor([0, 1, 2])
x.index_copy_(2, index, t)
print('[torch.Tensor.index_copy_ case-2]: ', x)

## case 3: assign case 1
x = torch.zeros(5, 3)
t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
index = torch.tensor([0, 4, 2])
y = x.index_copy_(0, index, t)
print('[torch.Tensor.index_copy_ case-3]: ', y)

## Case4: assign case 2
x = torch.zeros(2, 1, 3, 3)
t = torch.tensor([
    [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]],
    [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]], dtype=torch.float)
index = torch.tensor([0, 1, 2])
y = x.index_copy_(2, index, t)
print('[torch.Tensor.index_copy_ case-4]: ', y)


## Case5: scalar and assign
x = torch.zeros(20)
t = torch.tensor([1,3,4,5], dtype=torch.float)
index = torch.tensor([0, 12, 2, 1])
y = x.index_copy_(0, index, t)
print('[torch.Tensor.index_copy_ case-5]: ', y)

# torch.Tensor.requires_grad
## Case1:
x = torch.tensor([[1., -1.], [1., 1.]], requires_grad=True)
print('[torch.Tensor.requires_grad case-1]: ', x.requires_grad)

# torch.Tensor.to
## Case1:
tensor = torch.randn(2, 2)
print('[torch.Tensor.to case-1]: ', tensor.to(torch.float64))

## Case2:
tensor = torch.randn(2, 2)
print('[torch.Tensor.to case-2]: ', tensor.to(tensor.dtype))

## Case3:
tensor = torch.randn(2, 2)
print('[torch.Tensor.to case-3]: ', tensor.to(torch.float32))

## Case4:
tensor = torch.randn(2, 2)
print('[torch.Tensor.to case-4]: ', tensor.to(torch.float16))

# Case5:
tensor = torch.randn(2, 2)
print('[torch.Tensor.to case-5]: ', tensor.to(torch.int32))


# torch.utils.data.BatchSampler
## Case1: must 3 parameters 
x = torch.utils.data.BatchSampler(bs, batch_size=3, drop_last=False)
print('[torch.utils.data.BatchSampler case-1]: ', list(x))

# torch.Generator()
## Case1: default cpu
g_cpu = torch.Generator()
print('[torch.Generator() case-1]: ', g_cpu)

## Case2: cpu
g_cpu = torch.Generator(device='cpu')
print('[torch.Generator() case-2]: ', g_cpu)

## Case3: cpu
g_cpu = torch.Generator('cpu')
print('[torch.Generator() case-3]: ', g_cpu)

## Case4: cuda
g_cuda = torch.Generator('cuda')
print('[torch.Generator() case-4]: ', g_cuda)

## Case5: cuda
g_cuda = torch.Generator(device='cuda')
print('[torch.Generator() case-5]: ', g_cuda)

# torch.cdist
## Case1: 2d data and p<25
a = torch.tensor([[0.9041,  0.0196], [-0.3108, -2.4423], [-0.4821,  1.059]])
b = torch.tensor([[-2.1763, -0.4713], [-0.6986,  1.3702]])
c = torch.cdist(a, b, p=2)
print('[torch.cdist case-1]: ', c)

## Case2: 3d data and p<25 
a = torch.tensor([
    [[0.9041,  0.0196], [-0.3108, -2.4423], [-0.4821,  1.059]],
    [[0.9041,  0.0196], [-0.3108, -2.4423], [-0.4821,  1.059]]])
b = torch.tensor([
    [-2.1763, -0.4713], [-0.6986,  1.3702],
    [-2.1763, -0.4713], [-0.6986,  1.3702]])
c = torch.cdist(a, b, p=2)
print('[torch.cdist case-2]: ', c)

## Case3: 3d and 2d data 
a = torch.tensor([
    [[0.9041,  0.0196], [-0.3108, -2.4423], [-0.4821,  1.059]],
    [[0.9041,  0.0196], [-0.3108, -2.4423], [-0.4821,  1.059]]])
b = torch.tensor([[-2.1763, -0.4713], [-0.6986,  1.3702]])
c = torch.cdist(b, a, p=2)
print('[torch.cdist case-3]: ', c)

## Case4: 2d and 3d data 
a = torch.tensor([[0.9041,  0.0196], [-0.3108, -2.4423], [-0.4821,  1.059]])
b = torch.tensor([
    [[-2.1763, -0.4713], [-0.6986,  1.3702]],
    [[-2.1763, -0.4713], [-0.6986,  1.3702]]])
c = torch.cdist(a, b, p=2)
print('[torch.cdist case-4]: ', c)

# Case5: compute_mode parameteruse P>25 not influents
a = torch.tensor([[[0.9041,  0.0196], [-0.3108, -2.4423], [-0.4821,  1.059],
    [0.9041,  0.0196], [-0.3108, -2.4423], [-0.4821,  1.059],
    [0.9041,  0.0196], [-0.3108, -2.4423], [-0.4821,  1.059],
    [0.9041,  0.0196], [-0.3108, -2.4423], [-0.4821,  1.059],
    [0.9041,  0.0196], [-0.3108, -2.4423], [-0.4821,  1.059],
    [0.9041,  0.0196], [-0.3108, -2.4423], [-0.4821,  1.059],
    [0.9041,  0.0196], [-0.3108, -2.4423], [-0.4821,  1.059],
    [0.9041,  0.0196], [-0.3108, -2.4423], [-0.4821,  1.059],
    [0.9041,  0.0196], [-0.3108, -2.4423], [-0.4821,  1.059],
    [0.9041,  0.0196], [-0.3108, -2.4423], [-0.4821,  1.059],
    [0.9041,  0.0196], [-0.3108, -2.4423], [-0.4821,  1.059]]])
b = torch.tensor([[-2.1763, -0.4713], [-0.6986,  1.3702]])
c = torch.cdist(a, b, p=2,compute_mode='use_mm_for_euclid_dist_if_necessary')
print('[torch.cdist case-5]: ', c)

## Case4: 2d and 3d data 
a = torch.tensor([[0.9041,  0.0196], [-0.3108, -2.4423], [-0.4821,  1.059]])
b = torch.tensor([
    [[-2.1763, -0.4713], [-0.6986,  1.3702]],
    [[-2.1763, -0.4713], [-0.6986,  1.3702]]])
c = torch.cdist(a, b, p=2)
print('[torch.cdist case-4]: ', c)

# Case5: compute_mode parameteruse P>25 not influents
a = torch.tensor([[[0.9041,  0.0196], [-0.3108, -2.4423], [-0.4821,  1.059],
    [0.9041,  0.0196], [-0.3108, -2.4423], [-0.4821,  1.059],
    [0.9041,  0.0196], [-0.3108, -2.4423], [-0.4821,  1.059],
    [0.9041,  0.0196], [-0.3108, -2.4423], [-0.4821,  1.059],
    [0.9041,  0.0196], [-0.3108, -2.4423], [-0.4821,  1.059],
    [0.9041,  0.0196], [-0.3108, -2.4423], [-0.4821,  1.059],
    [0.9041,  0.0196], [-0.3108, -2.4423], [-0.4821,  1.059],
    [0.9041,  0.0196], [-0.3108, -2.4423], [-0.4821,  1.059],
    [0.9041,  0.0196], [-0.3108, -2.4423], [-0.4821,  1.059],
    [0.9041,  0.0196], [-0.3108, -2.4423], [-0.4821,  1.059],
    [0.9041,  0.0196], [-0.3108, -2.4423], [-0.4821,  1.059]]])
b = torch.tensor([[-2.1763, -0.4713], [-0.6986,  1.3702]])
c = torch.cdist(a, b, p=2,compute_mode='use_mm_for_euclid_dist_if_necessary')
print('[torch.cdist case-5]: ', c)

# torch.nn.InstanceNorm3d
## Case1: 5d input data With Learnable Parameters
m = torch.nn.InstanceNorm3d(2, affine=True)
input = b = torch.tensor([[[
    [[-2.1763, -0.4713], [-0.6986,  1.3702]],
    [[-2.1763, -0.4713], [-0.6986,  1.3702]],
    [[-2.1763, -0.4713], [-0.6986,  1.3702]],
    [[-2.1763, -0.4713], [-0.6986,  1.3702]]],
    [[[-2.1763, -0.4713], [-0.6986,  1.3702]],
    [[-2.1763, -0.4713], [-0.6986,  1.3702]],
    [[-2.1763, -0.4713], [-0.6986,  1.3702]],
    [[-2.1763, -0.4713], [-0.6986,  1.3702]]]]])
output = m(input)
print('[torch.nn.InstanceNorm3d case-1]: ', output)

# torch.nn.InstanceNorm3d
## Case2: 5d input data With not Learnable Parameters
m = torch.nn.InstanceNorm3d(2, affine=False)
input = b = torch.tensor([[[
    [[-2.1763, -0.4713], [-0.6986,  1.3702]],
    [[-2.1763, -0.4713], [-0.6986,  1.3702]],
    [[-2.1763, -0.4713], [-0.6986,  1.3702]],
    [[-2.1763, -0.4713], [-0.6986,  1.3702]]],
    [[[-2.1763, -0.4713], [-0.6986,  1.3702]],
    [[-2.1763, -0.4713], [-0.6986,  1.3702]],
    [[-2.1763, -0.4713], [-0.6986,  1.3702]],
    [[-2.1763, -0.4713], [-0.6986,  1.3702]]]]])
output = m(input)
print('[torch.nn.InstanceNorm3d case-2]: ', output)

## Case3: 4d input data With Learnable Parameters
# for 4d data api can be translated but m(input) can't accept 4d data, then user deal with it 
m = torch.nn.InstanceNorm3d(2, affine=True)
input = b = torch.tensor([[
    [[-2.1763, -0.4713], [-0.6986,  1.3702]],
    [[-2.1763, -0.4713], [-0.6986,  1.3702]],
    [[-2.1763, -0.4713], [-0.6986,  1.3702]],
    [[-2.1763, -0.4713], [-0.6986,  1.3702]]],
    [[[-2.1763, -0.4713], [-0.6986,  1.3702]],
    [[-2.1763, -0.4713], [-0.6986,  1.3702]],
    [[-2.1763, -0.4713], [-0.6986,  1.3702]],
    [[-2.1763, -0.4713], [-0.6986,  1.3702]]]])
# output = m(input)
print('[torch.nn.InstanceNorm3d case-3]: ', 'gg')

# torch.nn.InstanceNorm3d
## Case4: 4d input data With not Learnable Parameters
# for 4d data api can be translated but m(input) can't accept 4d data, then user deal with it 
m = torch.nn.InstanceNorm3d(2, affine=False)
input = b = torch.tensor([[
    [[-2.1763, -0.4713], [-0.6986,  1.3702]],
    [[-2.1763, -0.4713], [-0.6986,  1.3702]],
    [[-2.1763, -0.4713], [-0.6986,  1.3702]],
    [[-2.1763, -0.4713], [-0.6986,  1.3702]]],
    [[[-2.1763, -0.4713], [-0.6986,  1.3702]],
    [[-2.1763, -0.4713], [-0.6986,  1.3702]],
    [[-2.1763, -0.4713], [-0.6986,  1.3702]],
    [[-2.1763, -0.4713], [-0.6986,  1.3702]]]])
# output = m(input)
print('[torch.nn.InstanceNorm3d case-4]: ', 'gg')

## Case5: 5d input data With Learnable Parameters 
m = torch.nn.InstanceNorm3d(2, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
input = b = torch.tensor([[[
    [[-2.1763, -0.4713], [-0.6986,  1.3702]],
    [[-2.1763, -0.4713], [-0.6986,  1.3702]],
    [[-2.1763, -0.4713], [-0.6986,  1.3702]],
    [[-2.1763, -0.4713], [-0.6986,  1.3702]]],
    [[[-2.1763, -0.4713], [-0.6986,  1.3702]],
    [[-2.1763, -0.4713], [-0.6986,  1.3702]],
    [[-2.1763, -0.4713], [-0.6986,  1.3702]],
    [[-2.1763, -0.4713], [-0.6986,  1.3702]]]]])
output = m(input)
print('[torch.nn.InstanceNorm3d case-4]: ', output)

# torch.nn.BCEWithLogitsLoss
## Case1: torch demo
target = torch.ones([10, 64], dtype=torch.float32)
output = torch.full([10, 64], 1.5)  
pos_weight = torch.ones([64])
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
output = criterion(output, target)  
print('[torch.nn.BCEWithLogitsLoss case-1]: ', output)

# torch.nn.BCEWithLogitsLoss
## Case2: 
target = torch.ones([10, 64], dtype=torch.float32)
output = torch.full([10, 64], 1.5)  
pos_weight = torch.ones([64])
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight,size_average=True)
output = criterion(output, target)  
print('[torch.nn.BCEWithLogitsLoss case-2]: ', output)

# torch.nn.BCEWithLogitsLoss
## Case1: 
target = torch.ones([10, 64], dtype=torch.float32)
output = torch.full([10, 64], 1.5)  
pos_weight = torch.ones([64])
criterion = torch.nn.BCEWithLogitsLoss(reduce=False, size_average=False, reduction='none')
output = criterion(output, target)  
print('[torch.nn.BCEWithLogitsLoss case-3]: ', output)

# torch.nn.BCEWithLogitsLoss
## Case1: 
target = torch.ones([10, 64], dtype=torch.float32)
output = torch.full([10, 64], 1.5)  
pos_weight = torch.ones([64])
criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
output = criterion(output, target)  
print('[torch.nn.BCEWithLogitsLoss case-4]: ', output)

# torch.nn.BCEWithLogitsLoss
## Case1: 
target = torch.ones([10, 64], dtype=torch.float32)
output = torch.full([10, 64], 1.5)  
pos_weight = torch.ones([64])
criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
output = criterion(output, target)  
print('[torch.nn.BCEWithLogitsLoss case-5]: ', output)



