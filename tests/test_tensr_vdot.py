
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap

from tests.apibase import APIBase

obj = APIBase('torch.Tensor.vdot')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([4., 9., 3.])
        b = torch.tensor([4., 9., 3.])
        result = a.vdot(b)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([4., 9., 3.])
        result = a.vdot(torch.tensor([4., 9., 3.]))
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([4., 9., 3.])
        b = torch.tensor([4., 9., 3.])
        result = a.vdot(other=b)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([4., 9., 3.])
        b = torch.tensor([4., 9., 3.])
        result = 3. * a.vdot(other=b)
        '''
    )
    obj.run(pytorch_code, ['result'])