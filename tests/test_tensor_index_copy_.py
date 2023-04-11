
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap
from tests.apibase import APIBase


obj = APIBase('torch.Tensor.index_copy_')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x = torch.zeros(5, 3)
        t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
        index = torch.tensor([0, 4, 2])
        result = x.index_copy_(0, index, t)
        '''
    )
    obj.run(pytorch_code,['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x = torch.zeros(2, 1, 3, 3)
        t = torch.tensor([
            [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]],
            [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]], dtype=torch.float)
        index = torch.tensor([0, 1, 2])
        result = x.index_copy_(2, index, t)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x = torch.zeros(5, 3)
        t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
        index = torch.tensor([2, 2, 2])
        result = x.index_copy_(0, index, t)
        '''
    )
    obj.run(pytorch_code, ['result'])



def test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x = torch.zeros(2, 1, 3, 3)
        t = torch.tensor([
            [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]],
            [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]], dtype=torch.float)
        index = torch.tensor([0, 1, 2])
        result = x.index_copy_(2, index, t)
        '''
    )
    obj.run(pytorch_code, ['result'])


def test_case_5():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x = torch.zeros(20)
        t = torch.tensor([1,3,4,5], dtype=torch.float)
        index = torch.tensor([0, 12, 2, 1])
        result = x.index_copy_(0, index, t)
        '''
    )
    obj.run(pytorch_code, ['result'])


