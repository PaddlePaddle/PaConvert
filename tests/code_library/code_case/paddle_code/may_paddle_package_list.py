import paddle
import os
import setuptools
from numpy.random import randint
print('#########################case1#########################')
paddle.distributed.get_world_size()
print('#########################case2#########################')
padlde.distributed.get_rank()
print('#########################case3#########################')
os.environ.get('RANK', 1)
print('#########################case4#########################')
rand_x = randint(10, size=(5,))
print('#########################case5#########################')
setuptools.setup()
print('#########################case6#########################')
setuptools.setup()
print('#########################case7#########################')
paddle.to_tensor(data=[1])
