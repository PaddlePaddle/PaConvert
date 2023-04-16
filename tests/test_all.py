
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap

from tests.apibase import APIBase

obj = APIBase('torch.all')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.rand(1, 2).bool()
        result = torch.all(a)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.rand(3, 4)
        result = torch.all(a)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.rand(4, 3)
        result = torch.all(a, 1)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.rand(4, 3)
        result = torch.all(a, 1, True)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_5():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.rand(4, 3)
        result = torch.all(a, dim=0, keepdim=False)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_6():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([True])
        torch.all(torch.tensor([[4, 0, 7], [0, 2, 6]]), dim=0, keepdim=False, out=a)
        '''
    )
    obj.run(pytorch_code, ['a'])