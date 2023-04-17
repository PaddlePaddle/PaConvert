import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')
import textwrap
from tests.apibase import APIBase


obj = APIBase('torch.erfc')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([1., 2., -3., -4., 5.])
        result = torch.erfc(a)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([[1., 2., -3., -4., 5.], [1., 2., -3., -4., 5.]])
        result = torch.erfc(input=a)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([1., 2., -3., -4., 5.])
        out = torch.tensor([1., 2., -3., -4., 5.])
        result = torch.erfc(a, out=out)
        '''
    )
    obj.run(pytorch_code, ['out'])

def test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.erfc(input=torch.tensor([1., 2., -3., -4., 5.]))
        '''
    )
    obj.run(pytorch_code, ['result'])
