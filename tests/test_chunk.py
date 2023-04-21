
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap

from tests.apibase import APIBase

obj = APIBase('torch.chunk')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x = torch.ones(2, 3)
        result = torch.chunk(x, 2)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.chunk(torch.ones(2, 3), chunks=2)
        '''
    )
    obj.run(pytorch_code, ['result'])

def _test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.chunk(torch.ones(2, 3), chunks=2, dim=1)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x = torch.arange(12)
        result = torch.chunk(x, chunks=6)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_5():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.chunk(torch.ones(4, 6), chunks=2, dim=0)
        '''
    )
    obj.run(pytorch_code, ['result'])