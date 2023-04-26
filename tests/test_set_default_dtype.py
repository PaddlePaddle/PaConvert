
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap

from tests.apibase import APIBase

obj = APIBase('torch.set_default_dtype')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        torch.set_default_dtype(torch.float64)
        result = torch.tensor([1.2, 3])
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        torch.set_default_dtype(torch.float64)
        result = torch.tensor([1.2, 3j])
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        torch.set_default_dtype(torch.float32)
        result = torch.tensor([1.2, 3.8])
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        torch.set_default_dtype(torch.float32)
        result = torch.tensor([1.2, 3j])
        '''
    )
    obj.run(pytorch_code, ['result'])