
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap
from tests.apibase import APIBase


obj = APIBase('torch.frac')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.frac(torch.tensor([1, 2.5, -3.2]))
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([1, 2.5, -3.2])
        result = torch.frac(a)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = [1, 2.5, -3.2]
        out = torch.tensor(a)
        result = torch.frac(torch.tensor(a), out=out)
        '''
    )
    obj.run(pytorch_code, ['result', 'out'])

def test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([1, 2.5, -3.2])
        result = torch.frac(input=a)
        '''
    )
    obj.run(pytorch_code, ['result'])