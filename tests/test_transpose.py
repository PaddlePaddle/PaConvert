import torch, six

import torch.nn

# torhc.Tensor.index_copy_
x.index_copy_(0, index, t)
x.index_copy_(1, index, source)

# torch.Tensor.to
tensor.to(torch.float64)

# torch.Tensor.requires_grad
x.requires_grad

# torch.utils.data.BatchSampler
BatchSampler(sampler, batch_size=3, drop_last=False)

# torch.nn.InstanceNorm3d
m = nn.InstanceNorm3d(100, affine=True)
m = nn.InstanceNorm3d(100, affine=False)

# torch.nn.BCEWithLogitsLoss
loss = nn.BCEWithLogitsLoss()
loss = torch.nn.BCEWithLogitsLoss(None, size_average=None, reduce=None, reduction='mean', pos_weight=None)

# torch.Generator()
# torch.cdist
# torch.Size