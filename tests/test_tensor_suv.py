
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')
import textwrap
from tests.apibase import APIBase


obj = APIBase('torch.Tensor.svd')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x = torch.tensor([[ 0.2364, -0.7752,  0.6372],
                        [ 1.7201,  0.7394, -0.0504],
                        [-0.3371, -1.0584,  0.5296],
                        [ 0.3550, -0.4022,  1.5569],
                        [ 0.2445, -0.0158,  1.1414]])
        result = x.svd()[1]
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x = torch.tensor([[ 0.2364, -0.7752,  0.6372],
                        [ 1.7201,  0.7394, -0.0504],
                        [-0.3371, -1.0584,  0.5296],
                        [ 0.3550, -0.4022,  1.5569],
                        [ 0.2445, -0.0158,  1.1414]])
        result = x.svd(some=False)[1]
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x = torch.tensor([[ 0.2364, -0.7752,  0.6372],
                        [ 1.7201,  0.7394, -0.0504],
                        [-0.3371, -1.0584,  0.5296],
                        [ 0.3550, -0.4022,  1.5569],
                        [ 0.2445, -0.0158,  1.1414]])
        result = x.svd(compute_uv=False)[1]
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x = torch.tensor([[ 0.2364, -0.7752,  0.6372],
                        [ 1.7201,  0.7394, -0.0504],
                        [-0.3371, -1.0584,  0.5296],
                        [ 0.3550, -0.4022,  1.5569],
                        [ 0.2445, -0.0158,  1.1414]])
        result = x.svd(some=False, compute_uv=False)[1]
        '''
    )
    obj.run(pytorch_code, ['result'])