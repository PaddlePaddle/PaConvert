
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap
from tests.apibase import APIBase


obj = APIBase('torch.multiply')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        input = torch.tensor([0.2015, -0.4255, 2.6087])
        other = torch.tensor([0.2015, -0.4255, 2.6087])
        result = torch.multiply(input, other)
        '''
    )
    obj.run(pytorch_code, ['result'])

def _test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        input = torch.tensor([0.2015, -0.4255,  2.6087])
        other = torch.tensor([2, 6, 4])
        result = torch.multiply(input, other)
        '''
    )
    obj.run(pytorch_code, ['result'])

def _test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        input = torch.tensor([0.2015, -0.4255,  2.6087])
        result = torch.multiply(input, other=5)
        '''
    )
    obj.run(pytorch_code, ['result'])

def _test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        input = torch.tensor([3, 6, 9])
        result = torch.multiply(input, other = torch.tensor([0.2015, -0.4255, 2.6087]))
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_5():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        input = torch.tensor([0.2015, -0.4255,  2.6087])
        out = torch.tensor([0.2015, -0.4255,  2.6087])
        result = torch.multiply(input, torch.tensor([0.2015, -0.4255, 2.6087]), out=out)
        '''
    )
    obj.run(pytorch_code, ['result', 'out'])