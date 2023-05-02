
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap
from tests.apibase import APIBase


obj = APIBase('torch.lgamma')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        input = torch.tensor([0.34, 1.5, 0.73])
        result = torch.lgamma(input)
        '''
    )
    obj.run(pytorch_code,['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.lgamma(torch.tensor([0.34, 1.5, 0.73]))
        '''
    )
    obj.run(pytorch_code,['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        input = torch.tensor([0.34, 1.5, 0.73])
        result = torch.lgamma(input=input)
        '''
    )
    obj.run(pytorch_code,['result'])

def test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        input = torch.tensor([0.34, 1.5, 0.73])
        out = torch.tensor([0.34, 1.5, 0.73])
        result = torch.lgamma(input, out=out)
        '''
    )
    obj.run(pytorch_code,['result', 'out'])