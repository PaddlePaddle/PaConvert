
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap

from tests.apibase import APIBase

obj = APIBase('torch.tile')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x = torch.tensor([1, 2, 3])
        result = torch.tile(x, (2,))
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x = torch.tensor([[1, 2], [0, 6]])
        result = torch.tile(x, (2, 3))
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x = torch.zeros(8, 6, 4, 2)
        result = torch.tile(x, (2, 2))
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x = torch.zeros(4, 2)
        result = torch.tile(x, dims=(3, 3, 2, 2))
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_5():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x = torch.zeros(4, 2)
        dim = (3, 3, 2, 2)
        result = torch.tile(x, dims=dim)
        '''
    )
    obj.run(pytorch_code, ['result'])