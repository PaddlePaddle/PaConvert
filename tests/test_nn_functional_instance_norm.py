import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap
import numpy as np
from tests.apibase import APIBase


obj = APIBase('torch.nn.functional.instance_norm')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch.nn.functional as F
        import torch
        input = torch.tensor([[[ 1.1524,  0.4714,  0.2857],
         [-1.2533, -0.9829, -1.0981],
         [ 0.1507, -1.1431, -2.0361]],

        [[ 0.1024, -0.4482,  0.4137],
         [ 0.9385,  0.4565,  0.7702],
         [ 0.4135, -0.2587,  0.0482]]])
        data = torch.tensor([1., 1., 1.])
        result = F.instance_norm(input, data, data, data, data)
        '''
    )
    obj.run(pytorch_code,['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch.nn.functional as F
        import torch
        input = torch.tensor([[[ 1.1524,  0.4714,  0.2857],
         [-1.2533, -0.9829, -1.0981],
         [ 0.1507, -1.1431, -2.0361]],

        [[ 0.1024, -0.4482,  0.4137],
         [ 0.9385,  0.4565,  0.7702],
         [ 0.4135, -0.2587,  0.0482]]])
        data = torch.tensor([1., 1., 1.])
        result = F.instance_norm(input=input, running_mean=data, running_var=data, weight=data, bias=data)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch.nn.functional as F
        import torch
        input = torch.tensor([[[ 1.1524,  0.4714,  0.2857],
         [-1.2533, -0.9829, -1.0981],
         [ 0.1507, -1.1431, -2.0361]],

        [[ 0.1024, -0.4482,  0.4137],
         [ 0.9385,  0.4565,  0.7702],
         [ 0.4135, -0.2587,  0.0482]]])
        data = torch.tensor([1., 1., 1.])
        result = F.instance_norm(input=input, running_mean=data, running_var=data, weight=data, bias=data, momentum=0.5)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch.nn.functional as F
        import torch
        input = torch.tensor([[[ 1.1524,  0.4714,  0.2857],
         [-1.2533, -0.9829, -1.0981],
         [ 0.1507, -1.1431, -2.0361]],

        [[ 0.1024, -0.4482,  0.4137],
         [ 0.9385,  0.4565,  0.7702],
         [ 0.4135, -0.2587,  0.0482]]])
        data = torch.tensor([1., 1., 1.])
        result = F.instance_norm(input=input, running_mean=data, running_var=data, weight=data, bias=data, eps=1e-4)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_5():
    pytorch_code = textwrap.dedent(
        '''
        import torch.nn.functional as F
        import torch
        input = torch.tensor([[[ 1.1524,  0.4714,  0.2857],
         [-1.2533, -0.9829, -1.0981],
         [ 0.1507, -1.1431, -2.0361]],

        [[ 0.1024, -0.4482,  0.4137],
         [ 0.9385,  0.4565,  0.7702],
         [ 0.4135, -0.2587,  0.0482]]])
        data = torch.tensor([1., 1., 1.])
        result = F.instance_norm(input=input, running_mean=data, running_var=data, weight=data, bias=data, eps=1e-4, use_input_stats=True)
        '''
    )
    obj.run(pytorch_code, ['result'])