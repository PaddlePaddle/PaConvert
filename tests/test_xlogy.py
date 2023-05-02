import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')
import textwrap
from tests.apibase import APIBase


obj = APIBase('torch.xlogy')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([1., 2., 3., 4., 5.])
        b = torch.tensor([1., 2., 3., 4., 5.])
        result = torch.xlogy(a, b)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.xlogy(input=torch.tensor([1., 2., 3., 4., 5.]), other=torch.tensor([1., 2., 3., 4., 5.]))
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([1., 2., 3., 4., 5.])
        out = torch.tensor([1., 2., 3., 4., 5.])
        result = torch.xlogy(a, a, out=out)
        '''
    )
    obj.run(pytorch_code, ['out'])

def test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.xlogy(torch.tensor([1.]),torch.tensor([1., 2., 3., 4., 5.]))
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_5():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.xlogy(1.,torch.tensor([1., 2., 3., 4., 5.]))
        '''
    )
    obj.run(pytorch_code, ['result'])

def _test_case_6():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.xlogy(1, torch.tensor([1, 2, 3, 4, 5]))
        '''
    )
    obj.run(pytorch_code, ['result'])

def _test_case_7():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x = torch.tensor([1., float('inf'), float('nan')])
        result = torch.xlogy(0, x)
        '''
    )
    obj.run(pytorch_code, ['result'])

def _test_case_8():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.xlogy(1, torch.tensor([1., 2., 3., 4., 5.], dtype=torch.float64))
        '''
    )
    obj.run(pytorch_code, ['result'])