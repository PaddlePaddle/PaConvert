
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap
from tests.apibase import APIBase


obj = APIBase('torch.nn.functional.softmin')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        import torch.nn as nn
        input = torch.tensor([[-1.2837, -0.0297,  0.0355],
            [ 0.9112, -1.7526, -0.4061]])
        result = torch.nn.functional.softmin(input, dim=1)
        '''
    )
    obj.run(pytorch_code,['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        import torch.nn.functional as F
        input = torch.tensor([[-1.2837, -0.0297,  0.0355],
            [ 0.9112, -1.7526, -0.4061]])
        result = F.softmin(input=input, dim=1)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        import torch.nn.functional as F
        input = torch.tensor([[-1.2837, -0.0297,  0.0355],
            [ 0.9112, -1.7526, -0.4061]])
        result = F.softmin(input=input, dim=-1, dtype=torch.float64)
        '''
    )
    obj.run(pytorch_code, ['result'])