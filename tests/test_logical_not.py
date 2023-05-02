
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap
from tests.apibase import APIBase


obj = APIBase('torch.logical_not')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.logical_not(torch.tensor([True, False, True]))
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([0, 1, 10, 0], dtype=torch.int8)
        result = torch.logical_not(a)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([0, 1, 10, 0], dtype=torch.float32)
        result = torch.logical_not(a)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([0, 1, 10, 0], dtype=torch.float32)
        result = torch.logical_not(input=torch.tensor([0, 1, 10., 0.]))
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_5():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([0, 1, 10, 0], dtype=torch.float32)
        out = torch.tensor([True, False, True, True])
        result = torch.logical_not(a, out=out)
        '''
    )
    obj.run(pytorch_code, ['result', 'out'])