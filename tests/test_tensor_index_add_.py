import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')
import textwrap
from tests.apibase import APIBase


obj = APIBase('torch.Tensor.index_add_')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x= torch.ones([5, 3])
        t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
        index = torch.tensor([0, 4, 2])
        x.index_add_(0, index, t)
        '''
    )
    obj.run(pytorch_code, ['x'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x= torch.ones([5, 3])
        t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
        index = torch.tensor([0, 4, 2])
        x.index_add_(dim=0, index=index, source=t)
        '''
    )
    obj.run(pytorch_code, ['x'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x= torch.ones([5, 3])
        t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
        index = torch.tensor([0, 4, 2])
        x.index_add_(dim=0, index=index, source=t, alpha=3)
        '''
    )
    obj.run(pytorch_code, ['x'])

def test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x= torch.ones([5, 3])
        t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
        index = torch.tensor([0, 4, 2])
        x.index_add_(dim=0, index=index, source=t, alpha=-1)
        '''
    )
    obj.run(pytorch_code, ['x'])
