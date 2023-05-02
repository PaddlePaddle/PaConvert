
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap
from tests.apibase import APIBase


obj = APIBase('torch.logical_or')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.logical_or(torch.tensor([True, False, True]), torch.tensor([True, False, False]))
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([0, 1, 10, 0], dtype=torch.int8)
        b = torch.tensor([4, 0, 1, 0], dtype=torch.int8)
        result = torch.logical_or(a, b)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([0, 1, 10, 0], dtype=torch.float32)
        b = torch.tensor([4, 0, 1, 0], dtype=torch.float32)
        result = torch.logical_or(a, b)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([0, 1, 10, 0], dtype=torch.float32)
        b = torch.tensor([4, 0, 1, 0], dtype=torch.float32)
        result = torch.logical_or(input=torch.tensor([0, 1, 10., 0.]), other=torch.tensor([4, 0, 10., 0.]))
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_5():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([0, 1, 10, 0], dtype=torch.float32)
        b = torch.tensor([4, 0, 1, 0], dtype=torch.float32)
        out = torch.tensor([True, False, True, True])
        result = torch.logical_or(a, b, out=out)
        '''
    )
    obj.run(pytorch_code, ['result', 'out'])

def test_case_6():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([0, 1, 10, 0], dtype=torch.float32)
        b = torch.tensor([4, 0, 1, 0], dtype=torch.int8)
        result = torch.logical_or(a, b)
        '''
    )
    obj.run(pytorch_code, ['result'])