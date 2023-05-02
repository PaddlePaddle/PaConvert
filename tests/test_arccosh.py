
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap
from tests.apibase import APIBase


obj = APIBase('torch.arccosh')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.arccosh(torch.tensor([1.3192, 1.9915, 1.9674, 1.7151]))
        '''
    )
    obj.run(pytorch_code,['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([1.3192, 1.9915, 1.9674, 1.7151])
        result = torch.arccosh(a)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = [1.3192, 1.9915, 1.9674, 1.7151]
        out = torch.tensor(a)
        result = torch.arccosh(torch.tensor(a), out=out)
        '''
    )
    obj.run(pytorch_code, ['out', 'result'])
