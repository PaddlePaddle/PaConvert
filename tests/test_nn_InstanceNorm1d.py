import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap
import numpy as np
from tests.apibase import APIBase


obj = APIBase('torch.nn.InstanceNorm1d')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch.nn as nn
        import torch
        m = nn.InstanceNorm1d(3)
        input = torch.tensor([[[ 1.1524,  0.4714,  0.2857],
         [-1.2533, -0.9829, -1.0981],
         [ 0.1507, -1.1431, -2.0361]],

        [[ 0.1024, -0.4482,  0.4137],
         [ 0.9385,  0.4565,  0.7702],
         [ 0.4135, -0.2587,  0.0482]]])
        result = m(input)
        '''
    )
    obj.run(pytorch_code,['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch.nn as nn
        import torch
        m = nn.InstanceNorm1d(3, affine=True)
        input = torch.tensor([[[ 1.1524,  0.4714,  0.2857],
         [-1.2533, -0.9829, -1.0981],
         [ 0.1507, -1.1431, -2.0361]],

        [[ 0.1024, -0.4482,  0.4137],
         [ 0.9385,  0.4565,  0.7702],
         [ 0.4135, -0.2587,  0.0482]]])
        result = m(input)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch.nn as nn
        import torch
        m = nn.InstanceNorm1d(3, affine=False)
        input = torch.tensor([[[ 1.1524,  0.4714,  0.2857],
         [-1.2533, -0.9829, -1.0981],
         [ 0.1507, -1.1431, -2.0361]],

        [[ 0.1024, -0.4482,  0.4137],
         [ 0.9385,  0.4565,  0.7702],
         [ 0.4135, -0.2587,  0.0482]]])
        result = m(input)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch.nn as nn
        import torch
        m = nn.InstanceNorm1d(3, affine=True, momentum=0.1)
        input = torch.tensor([[[ 1.1524,  0.4714,  0.2857],
         [-1.2533, -0.9829, -1.0981],
         [ 0.1507, -1.1431, -2.0361]],

        [[ 0.1024, -0.4482,  0.4137],
         [ 0.9385,  0.4565,  0.7702],
         [ 0.4135, -0.2587,  0.0482]]])
        result = m(input)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_5():
    pytorch_code = textwrap.dedent(
        '''
        import torch.nn as nn
        import torch
        m = nn.InstanceNorm1d(3, affine=False, momentum=0.1)
        input = torch.tensor([[[ 1.1524,  0.4714,  0.2857],
         [-1.2533, -0.9829, -1.0981],
         [ 0.1507, -1.1431, -2.0361]],

        [[ 0.1024, -0.4482,  0.4137],
         [ 0.9385,  0.4565,  0.7702],
         [ 0.4135, -0.2587,  0.0482]]])
        result = m(input)
        '''
    )
    obj.run(pytorch_code, ['result'])