
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')
import textwrap
from tests.apibase import APIBase


obj = APIBase('torch.msort')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x = torch.tensor([[-1.3029,  0.4921, -0.7432],
                        [ 2.6672, -0.0987,  0.0750],
                        [ 0.1436, -1.0114,  1.3641]])
        result = torch.msort(x)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x = torch.tensor([[-1.3029,  0.4921, -0.7432],
                        [ 2.6672, -0.0987,  0.0750],
                        [ 0.1436, -1.0114,  1.3641]])
        result = torch.msort(input = x)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x = torch.tensor([[-1.3029,  0.4921, -0.7432],
                        [ 2.6672, -0.0987,  0.0750],
                        [ 0.1436, -1.0114,  1.3641]])
        torch.msort(x, out=x)
        '''
    )
    obj.run(pytorch_code, ['x'])