
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap
from tests.apibase import APIBase

obj = APIBase('torch.Size')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = list(torch.Size([2, 8, 64, 64]))
        '''
    )
    obj.run_bool(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.randn(6, 5, 7).size() == torch.Size([6, 5, 7])
        '''
    )
    obj.run_bool(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        out = torch.Size([6, 5, 7])
        result = out == torch.Size([6, 5, 7])
        '''
    )
    obj.run_bool(pytorch_code, ['result'])

def test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        data = torch.Size([1])
        result = list(data)
        '''
    )
    obj.run_bool(pytorch_code, ['result'])

def test_case_5():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        shape = torch.Size([1])
        result = list(shape)
        '''
    )
    obj.run_bool(pytorch_code, ['result'])