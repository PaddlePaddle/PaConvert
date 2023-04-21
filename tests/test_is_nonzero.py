
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')
import textwrap
from tests.apibase import APIBase


obj = APIBase('torch.is_nonzero')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.is_nonzero(torch.tensor([0.]))
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([1])
        result = torch.abs(a)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.is_nonzero(torch.tensor([False]))
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = [0]
        b = torch.tensor(a)
        result = torch.is_nonzero(b)
        '''
    )
    obj.run(pytorch_code, ['result'])