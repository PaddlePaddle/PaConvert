
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap

from tests.apibase import APIBase

obj = APIBase('torch.vdot')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([4., 9., 3.])
        b = torch.tensor([4., 9., 3.])
        result = torch.vdot(a, b)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([4., 9., 3.])
        result = torch.vdot(a, torch.tensor([4., 9., 3.]))
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([4., 9., 3.])
        b = torch.tensor([4., 9., 3.])
        result = torch.vdot(input=a, other=b)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([4., 9., 3.])
        b = torch.tensor([4., 9., 3.])
        out = torch.tensor([1.])
        result = torch.vdot(input=a, other=b, out=out)
        '''
    )
    obj.run(pytorch_code, ['out'])