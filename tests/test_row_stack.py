
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')
import textwrap
from tests.apibase import APIBase


obj = APIBase('torch.row_stack')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([4, 5, 6])
        result = torch.row_stack((a,b))
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([4, 5, 6])
        c = [a, b]
        result = torch.row_stack(c)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([[1],[2],[3]])
        b = torch.tensor([[4],[5],[6]])
        c = (a, b)
        result = torch.row_stack(c)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([4, 5, 6])
        out = torch.tensor([[4, 5, 6], [1, 2, 3]])
        result = torch.row_stack((a,b), out=out)
        '''
    )
    obj.run(pytorch_code, ['out'])

def test_case_5():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([[1, 2, 3]])
        b = torch.tensor([[4, 5, 6]])
        out = torch.tensor([[[4, 5, 6]], [[1, 2, 3]]])
        result = torch.row_stack((a,b), out=out)
        '''
    )
    obj.run(pytorch_code, ['out'])