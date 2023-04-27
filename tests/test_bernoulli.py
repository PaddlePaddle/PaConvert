
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap

from tests.apibase import APIBase

obj = APIBase('torch.bernoulli')

def _test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([0.8, 0.1, 0.4])
        result = torch.bernoulli(a)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.ones(3, 3)
        result = torch.bernoulli(a)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.bernoulli(torch.ones(3, 3))
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.bernoulli(torch.zeros(3, 3))
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_5():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.ones(3, 3)
        out = torch.ones(3, 3)
        result = torch.bernoulli(a, out=out)
        '''
    )
    obj.run(pytorch_code, ['result', 'out'])