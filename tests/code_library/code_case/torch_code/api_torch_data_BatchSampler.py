import torch
from torch.utils.data import BatchSampler

print("#########################case1#########################")
o = list(BatchSampler(range(10), batch_size=3, drop_last=True))
print("#########################case2#########################")
o = list(BatchSampler(range(10), batch_size=3, drop_last=False))
print("#########################case3#########################")
batch_sampler_train = torch.utils.data.BatchSampler(range(10), 2, drop_last=True)
print("#########################case4#########################")
batch_size = 4
batch_sampler_train = torch.utils.data.BatchSampler(
    range(10), batch_size, drop_last=False
)
print("#########################case5#########################")
batch_size = 4
batch_sampler_train = torch.utils.data.BatchSampler(
    sampler=range(10), batch_size=batch_size, drop_last=False
)
