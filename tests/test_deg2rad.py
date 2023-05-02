
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap
from tests.apibase import APIBase


obj = APIBase('torch.deg2rad')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([[180.0, -180.0], [360.0, -360.0], [90.0, -90.0]])
        result = torch.deg2rad(a)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.deg2rad(torch.tensor([[180.0, -180.0], [360.0, -360.0], [90.0, -90.0]]))
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([[180.0, -180.0], [360.0, -360.0], [90.0, -90.0]])
        out = torch.tensor([[180.0, -180.0], [360.0, -360.0], [90.0, -90.0]])
        result = torch.deg2rad(a, out=out)
        '''
    )
    obj.run(pytorch_code, ['result', 'out'])