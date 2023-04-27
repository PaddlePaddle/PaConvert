
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap

from tests.apibase import APIBase

obj = APIBase('torch.arange')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.arange(5)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.arange(5.)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.arange(1, 4)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.arange(1, 4, step=2)
        '''
    )
    obj.run(pytorch_code, ['result'])

def _test_case_5():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.arange(1, 2.5, 0.5)
        '''
    )
    obj.run(pytorch_code, ['result'])

def _test_case_6():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.arange(1, 2.5, 0.5, dtype=torch.float64, requires_grad=True)
        '''
    )
    obj.run(pytorch_code, ['result'])
