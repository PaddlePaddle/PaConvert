
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap
from tests.apibase import APIBase


obj = APIBase('torch.lerp')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        start = torch.tensor([1., 2., 3., 4.])
        end = torch.tensor([10., 10., 10., 10.])
        weight = torch.tensor([0.5, 1, 0.3, 0.6])
        result = torch.lerp(start, end, weight)
        '''
    )
    obj.run(pytorch_code,['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        weight = torch.tensor([0.5, 1, 0.3, 0.6])
        result = torch.lerp(torch.tensor([1., 2., 3., 4.]), torch.tensor([10., 10., 10., 10.]), weight)
        '''
    )
    obj.run(pytorch_code,['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        start = torch.tensor([1., 2., 3., 4.])
        end = torch.tensor([10., 10., 10., 10.])
        weight = torch.tensor([0.5, 1, 0.3, 0.6])
        result = torch.lerp(input=start, end=end, weight=weight)
        '''
    )
    obj.run(pytorch_code,['result'])

def test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        start = torch.tensor([1., 2., 3., 4.])
        end = torch.tensor([10., 10., 10., 10.])
        weight = torch.tensor([0.5, 1, 0.3, 0.6])
        out = torch.tensor([0.5, 1, 0.3, 0.6])
        result = torch.lerp(start, end, weight, out=out)
        '''
    )
    obj.run(pytorch_code,['result', 'out'])

def test_case_5():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        start = torch.tensor([1., 2., 3., 4.])
        end = torch.tensor([10., 10., 10., 10.])
        result = torch.lerp(input=start, end=end, weight=0.5)
        '''
    )
    obj.run(pytorch_code,['result'])