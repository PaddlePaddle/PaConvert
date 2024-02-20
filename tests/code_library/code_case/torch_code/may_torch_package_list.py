import torch
from os import environ
from numpy.random import randint
from setuptools import setup
import setuptools

print("#########################case1#########################")
environ.get("WORLD_SIZE",1)
print("#########################case2#########################")
os.environ.get('LOCAL_RANK',1)
print("#########################case3#########################")
os.environ.get('RANK',1)
print("#########################case4#########################")
rand_x=randint(10,size=(5,))
print('#########################case5#########################')
setup()
print('#########################case6#########################')
setuptools.setup()
print('#########################case7#########################')
torch.tensor([1])
