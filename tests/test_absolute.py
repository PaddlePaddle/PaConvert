
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap
from tests.apibase import APIBase


obj = APIBase('torch.absolute')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.absolute(torch.tensor([-1, -2, 3]))
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([-1, -2, 3])
        result = torch.absolute(a)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = [-1, -2, 3]
        out = torch.tensor(a)
        result = torch.absolute(torch.tensor(a), out=out)
        '''
    )
    obj.run(pytorch_code, ['out'])