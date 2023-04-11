
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')
import textwrap
from tests.apibase import APIBase


obj = APIBase('torch.set_printoptions')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        torch.set_printoptions(precision=2)
        '''
    )
    obj.run(pytorch_code)

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        torch.set_printoptions()
        '''
    )
    obj.run(pytorch_code)

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        torch.set_printoptions(precision=8, threshold=2000, edgeitems=4, linewidth=100, profile='full')
        '''
    )
    obj.run(pytorch_code)