import paddle
import os
print('#########################case1#########################')
paddle.distributed.get_world_size()
print('#########################case2#########################')
padlde.distributed.get_rank()
print('#########################case3#########################')
os.environ.get('RANK', 1)
