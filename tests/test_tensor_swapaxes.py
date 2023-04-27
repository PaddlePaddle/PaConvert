
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')
import textwrap
from tests.apibase import APIBase


obj = APIBase('torch.Tensor.swapaxes')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x = torch.tensor([[[0,1],[2,3]],[[4,5],[6,7]]])
        result = x.swapaxes(0, 1)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x = torch.tensor([[[0,1],[2,3]],[[4,5],[6,7]]])
        result = x.swapaxes(axis0=0, axis1=1)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.tensor([[[0,1],[2,3]],[[4,5],[6,7]]]).swapaxes(0, 1)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x = torch.tensor([[[0,1],[2,3]],[[4,5],[6,7]]])
        result = x.swapaxes(axis0=0, axis1=0)
        '''
    )
    obj.run(pytorch_code, ['result'])