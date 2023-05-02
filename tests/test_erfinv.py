
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap
from tests.apibase import APIBase


obj = APIBase('torch.erfinv')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.erfinv(torch.tensor([0, 0.5]))
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([0, 0.5])
        result = torch.erfinv(a)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([0, 0.5])
        out = torch.tensor([0, 0.5])
        result = torch.erfinv(a, out=out)
        '''
    )
    obj.run(pytorch_code, ['result', 'out'])

def test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([0, 0.5])
        result = torch.erfinv(input=a)
        '''
    )
    obj.run(pytorch_code, ['result'])
