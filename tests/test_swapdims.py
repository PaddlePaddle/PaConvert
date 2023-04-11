
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')
import textwrap
from tests.apibase import APIBase


obj = APIBase('torch.swapdims')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x = torch.tensor([[[0,1],[2,3]],[[4,5],[6,7]]])
        result = torch.swapdims(x, 0, 1)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x = torch.tensor([[[0,1],[2,3]],[[4,5],[6,7]]])
        result = torch.swapdims(input=x, dim0=0, dim1=1)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.swapdims(torch.tensor([[[0,1],[2,3]],[[4,5],[6,7]]]), 0, 1)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x = torch.tensor([[[0,1],[2,3]],[[4,5],[6,7]]])
        result = torch.swapdims(input=x, dim0=0, dim1=0)
        '''
    )
    obj.run(pytorch_code, ['result'])