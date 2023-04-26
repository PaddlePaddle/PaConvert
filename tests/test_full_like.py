
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap

from tests.apibase import APIBase

obj = APIBase('torch.full_like')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        input = torch.empty(2, 3)
        result = torch.full_like(input, 2)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        num = 5.
        result = torch.full_like(torch.empty(2, 3), num)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.full_like(torch.empty(2, 3), 10, dtype=torch.float64, requires_grad=True)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        flag = False
        result = torch.full_like(torch.empty(2, 3), fill_value=8., dtype=torch.float64, requires_grad=flag)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_5():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.full_like(torch.empty(2, 3), 6, layout=torch.strided, dtype=torch.float64, requires_grad=True)
        '''
    )
    obj.run(pytorch_code, ['result'])
