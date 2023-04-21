
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap

from tests.apibase import APIBase

obj = APIBase('torch.complex')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        real = torch.tensor([1, 2], dtype=torch.float32)
        imag = torch.tensor([3, 4], dtype=torch.float32)
        result = torch.complex(real, imag)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.complex(torch.tensor([1., 2]), torch.tensor([3., 4]))
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.complex(real=torch.tensor([1., 2.]), imag=torch.tensor([3., 4.]))
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.complex(torch.tensor([1., 2.], dtype=torch.float64), torch.tensor([3., 4.], dtype=torch.float64))
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_5():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        out = torch.tensor([2., 3.], dtype=torch.complex64)
        result = torch.complex(torch.tensor([1., 2]), torch.tensor([3., 4]), out=out)
        '''
    )
    obj.run(pytorch_code, ['result', 'out'])