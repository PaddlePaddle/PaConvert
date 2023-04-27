
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')
import textwrap
from tests.apibase import APIBase


obj = APIBase('torch.Tensor.narrow_copy')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = x.narrow_copy(0, 0, 2)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = x.narrow_copy(1, 1, 2)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = x.narrow_copy(-1, torch.tensor(-2), 1)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = x.narrow_copy(-1, torch.tensor(-1), length=1)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_5():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = x.narrow_copy(dim=-1, start=torch.tensor(-1), length=1)
        '''
    )
    obj.run(pytorch_code, ['result'])