
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap
from tests.apibase import APIBase


obj = APIBase('torch.erf')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.erf(torch.tensor([0, -1., 10.]))
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([0, -1., 10.])
        result = torch.erf(a)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([0, -1., 10.])
        out = torch.tensor([0, -1., 10.])
        result = torch.erf(a, out=out)
        '''
    )
    obj.run(pytorch_code, ['result', 'out'])

def test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([0, -1., 10.])
        result = torch.erf(input=a)
        '''
    )
    obj.run(pytorch_code, ['result'])