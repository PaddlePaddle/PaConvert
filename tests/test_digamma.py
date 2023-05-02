
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap
from tests.apibase import APIBase


obj = APIBase('torch.digamma')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.digamma(torch.tensor([1, 0.5]))
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([1, 0.5])
        result = torch.digamma(a)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([1, 0.5])
        out = torch.tensor([1, 0.5])
        result = torch.digamma(a, out=out)
        '''
    )
    obj.run(pytorch_code, ['result', 'out'])