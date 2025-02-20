import os

import einops.layers.paddle as einops_torch
import paddle
import setuptools
from einops.layers.paddle import Rearrange
from numpy.random import randint

print("#########################case1#########################")
paddle.distributed.get_world_size()
print("#########################case2#########################")
padlde.distributed.get_rank()
print("#########################case3#########################")
os.environ.get("RANK", 1)
print("#########################case4#########################")
rand_x = randint(10, size=(5,))
print("#########################case5#########################")
setuptools.setup()
print("#########################case6#########################")
setuptools.setup()
print("#########################case7#########################")
paddle.to_tensor(data=[1])
print("#########################case7#########################")
Rearrange("b (h w) -> b h w", h=16, w=16)
einops.layers.torch.Rearrange("b (h w) -> b h w", h=16, w=16)
einops_torch.Rearrange("b (h w) -> b h w", h=16, w=16)
