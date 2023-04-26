
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap

from tests.apibase import APIBase

obj = APIBase('torch.zeros_like')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        input = torch.empty(2, 3)
        result = torch.zeros_like(input)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.zeros_like(torch.empty(2, 3))
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.zeros_like(torch.empty(2, 3), dtype=torch.float64, requires_grad=True)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        flag = False
        result = torch.zeros_like(torch.empty(2, 3), dtype=torch.float64, requires_grad=flag)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_5():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.zeros_like(torch.empty(2, 3), layout=torch.strided, dtype=torch.float64, requires_grad=True)
        '''
    )
    obj.run(pytorch_code, ['result'])
