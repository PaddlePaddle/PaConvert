
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap
from tests.apibase import APIBase


obj = APIBase('torch.permute')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x = torch.tensor([[1, 2, 3], [3, 4, 6]])
        result = torch.permute(x, (1, 0))
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x = torch.tensor([[1, 2, 3], [3, 4, 6]])
        result = torch.permute(x, [1, 0])
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x = torch.tensor([[1, 2, 3], [3, 4, 6]])
        shape = [1, 0]
        result = torch.permute(x, shape)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x = torch.tensor([[1, 2, 3], [3, 4, 6]])
        a, b = 1, 0
        result = torch.permute(x, [a, b])
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_5():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x = torch.tensor([[[1, 2, 3], [3, 4, 6]]])
        result = torch.permute(input=x, dims=[1, 2, 0])
        '''
    )
    obj.run(pytorch_code, ['result'])