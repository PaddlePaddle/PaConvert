import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')
import textwrap
from tests.apibase import APIBase


obj = APIBase('torch.index_add')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x= torch.ones([5, 3])
        t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
        index = torch.tensor([0, 4, 2])
        result = torch.index_add(x, 0, index, t)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x= torch.ones([5, 3])
        t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
        index = torch.tensor([0, 4, 2])
        result = torch.index_add(input=x, dim=0, index=index, source=t)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x= torch.ones([5, 3])
        t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
        index = torch.tensor([0, 4, 2])
        result = torch.index_add(input=x, dim=0, index=index, source=t, alpha=3)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x= torch.ones([5, 3])
        t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
        index = torch.tensor([0, 4, 2])
        result = torch.index_add(input=x, dim=0, index=index, source=t, alpha=-1)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_5():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x= torch.ones([5, 3])
        t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
        index = torch.tensor([0, 4, 2])
        out = torch.ones([5, 3])
        result = torch.index_add(input=x, dim=0, index=index, source=t, alpha=3, out=out)
        '''
    )
    obj.run(pytorch_code, ['out'])
