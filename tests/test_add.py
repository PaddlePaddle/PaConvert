
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap

from tests.apibase import APIBase

obj = APIBase('torch.add')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.add(torch.tensor([1, 2, 3]), torch.tensor([1, 4, 6]))
        '''
    )
    obj.run(pytorch_code, ['result'], __file__)

# test_case_2 - ValueError: (InvalidArgument) add(): argument 'y' (position 1) must be Tensor, but got int (at ..\paddle\fluid\pybind\eager_utils.cc:894 
def _test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.add(torch.tensor([1, 2, 3]), 20)
        '''
    )
    obj.run(pytorch_code, ['result'], __file__)

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([20])
        result = torch.add(a, b)
        '''
    )
    obj.run(pytorch_code, ['result'], __file__)

def test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([20])
        result = torch.add(a, b, alpha = 10)
        '''
    )
    obj.run(pytorch_code, ['result'], __file__)

def test_case_5():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([1, 2, 3])
        result = torch.add(a, torch.tensor([1, 4, 6]), alpha = 10, out=a)
        '''
    )
    obj.run(pytorch_code, ['a'], __file__)
