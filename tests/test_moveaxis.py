
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap

from tests.apibase import APIBase

obj = APIBase('torch.moveaxis')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x = torch.arange(24)
        x = torch.reshape(x, (1, 4, 6))
        result = torch.moveaxis(x, 1, 0)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x = torch.arange(24)
        x = torch.reshape(x, (1, 4, 6))
        result = torch.moveaxis(x, (1, 0), (0, 1))
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x = torch.arange(24)
        x = torch.reshape(x, (1, 4, 6))
        a, b = 0, 1
        result = torch.moveaxis(x, (a, b), (b, a))
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x = torch.arange(24)
        x = torch.reshape(x, (1, 4, 6))
        a, b = [0, 1], [1, 0]
        result = torch.moveaxis(x, a, b)
        '''
    )
    obj.run(pytorch_code, ['result'])

