
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap

from tests.apibase import APIBase

obj = APIBase('torch.eye')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.eye(3)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.eye(3, 5)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.eye(3, 5, layout=torch.strided, requires_grad=True)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.eye(n=3, m=3, dtype=torch.int64)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_5():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = 3
        out = torch.tensor([2., 3.])
        result = torch.eye(a, a, out=out)
        '''
    )
    obj.run(pytorch_code, ['result', 'out'])




