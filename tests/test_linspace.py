
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap

from tests.apibase import APIBase

obj = APIBase('torch.linspace')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.linspace(3, 10, 5)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.linspace(-10., 10., 5)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.linspace(start=1, end=10, steps=4)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.linspace(1, 4, 2, layout=torch.strided, requires_grad=True)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_5():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.linspace(-1, 3, 4, dtype=torch.float64)
        '''
    )
    obj.run(pytorch_code, ['result'])


def test_case_6():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        out = torch.tensor([1., 2, 3], dtype=torch.float64)
        result = torch.linspace(-1, 3, 4, dtype=torch.float64, out=out)
        '''
    )
    obj.run(pytorch_code, ['result', 'out'])