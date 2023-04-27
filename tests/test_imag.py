
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap

from tests.apibase import APIBase

obj = APIBase('torch.imag')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x = torch.tensor([4+3j], dtype=torch.cfloat)
        result = torch.imag(x)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.imag(torch.tensor([4+9j, 6+0.9j], dtype=torch.cfloat))
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.imag(torch.tensor([6, 9.], dtype=torch.complex128))
        '''
    )
    obj.run(pytorch_code, ['result'])