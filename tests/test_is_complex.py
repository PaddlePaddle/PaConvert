
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap

from tests.apibase import APIBase

obj = APIBase('torch.is_tensor')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([[4, 9], [23, 2]])
        result = torch.is_complex(a)
        '''
    )
    obj.run_bool(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.is_complex(torch.tensor([[4, 9], [23, 2]], dtype=torch.complex64))
        '''
    )
    obj.run_bool(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([[4, 9], [23, 2]], dtype=torch.complex128)
        result = torch.is_complex(a)
        '''
    )
    obj.run_bool(pytorch_code, ['result'])