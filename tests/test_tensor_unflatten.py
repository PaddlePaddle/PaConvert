import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')
import textwrap
from tests.apibase import APIBase


obj = APIBase('torch.Tensor.unflatten')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([[ 0.2180,  1.0558,  0.1608,  0.9245],
                [ 1.3794,  1.4090,  0.2514, -0.8818],
                [-0.4561,  0.5123,  1.7505, -0.4094]])
        result = a.unflatten(-1, (2, 2))
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([[ 0.2180,  1.0558,  0.1608,  0.9245],
                [ 1.3794,  1.4090,  0.2514, -0.8818],
                [-0.4561,  0.5123,  1.7505, -0.4094]])
        result = a.unflatten(1, (2, 2))
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([[ 0.2180,  1.0558,  0.1608,  0.9245],
                [ 1.3794,  1.4090,  0.2514, -0.8818],
                [-0.4561,  0.5123,  1.7505, -0.4094]])
        result = a.unflatten(-1, (2, 1, 2))
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([[ 0.2180,  1.0558,  0.1608,  0.9245],
                [ 1.3794,  1.4090,  0.2514, -0.8818],
                [-0.4561,  0.5123,  1.7505, -0.4094]])
        result = a.unflatten(dim=-1, sizes=(2, 1, 2))
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_5():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([[ 0.2180,  1.0558,  0.1608,  0.9245],
                [ 1.3794,  1.4090,  0.2514, -0.8818],
                [-0.4561,  0.5123,  1.7505, -0.4094]])
        result = a.unflatten(dim=-1, sizes=[2, 1, 2])
        '''
    )
    obj.run(pytorch_code, ['result'])
