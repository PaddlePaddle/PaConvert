import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')
import textwrap
from tests.apibase import APIBase


obj = APIBase('torch.Tensor.addr')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([4., 5., 6.])
        b = torch.tensor([1., 2., 3.])
        input = torch.tensor([1., 2., 3.])
        result = input.addr(a, b)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([4., 5., 6.])
        b = torch.tensor([1., 2., 3.])
        input = torch.tensor([1., 2., 3.])
        result = input.addr(a, b, beta=3)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([4., 5., 6.])
        b = torch.tensor([1., 2., 3.])
        input = torch.tensor([1., 2., 3.])
        result = input.addr(vec1=a, vec2=b, beta=3, alpha=3)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        input = torch.tensor([1., 2., 3.])
        result = input.addr(vec1=torch.tensor([4., 5., 6.]), vec2=torch.tensor([1., 2., 3.]), beta=3, alpha=3)
        '''
    )
    obj.run(pytorch_code, ['result'])