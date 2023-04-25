import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')
import textwrap
from tests.apibase import APIBase


obj = APIBase('torch.Tensor.logaddexp2')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([1., 2., -3., -4., 5.])
        b = torch.tensor([1., 2., -3., -4., 5.])
        result = a.logaddexp2(b)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.tensor([1., 2., -3., -4., 5.]).logaddexp2(other=torch.tensor([1., 2., -3., -4., 5.]))
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([1., 2., -3., -4., 5.])
        result = 2 * a.logaddexp2(a) + 1
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.tensor([1.]).logaddexp2(torch.tensor([1., 2., -3., -4., 5.]))
        '''
    )
    obj.run(pytorch_code, ['result'])
