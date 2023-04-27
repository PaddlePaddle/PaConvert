
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap

from tests.apibase import APIBase

obj = APIBase('torch.is_floating_point')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([[4, 9], [23, 2]], dtype=torch.int64)
        result = torch.is_floating_point(a)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([[4, 9], [23, 2]], dtype=torch.float64)
        result = torch.is_floating_point(a)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.is_floating_point(torch.tensor([[4, 9], [23, 2]], dtype=torch.float32))
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.is_floating_point(torch.tensor([[4, 9], [23, 2]], dtype=torch.float16))
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_5():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.is_floating_point(torch.tensor([[4, 9], [23, 2]], dtype=torch.bfloat16))
        '''
    )
    obj.run(pytorch_code, ['result'])