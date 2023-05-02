
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap
from tests.apibase import APIBase


obj = APIBase('torch.logit')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        input = torch.tensor([0.2796, 0.9331, 0.6486, 0.1523, 0.6516])
        result = torch.logit(input, eps=1e-6)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        input = torch.tensor([0.2796, 0.9331, 0.6486, 0.1523, 0.6516])
        eps = 1e-6
        result = torch.logit(input, eps)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.logit(torch.tensor([0.2796, 0.9331, 0.6486, 0.1523, 0.6516]), eps=1e-6)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        input = torch.tensor([0.2796, 0.9331, 0.6486, 0.1523, 0.6516])
        out = torch.tensor([0.2796, 0.9331, 0.6486, 0.1523, 0.6516])
        result = torch.logit(input, eps=1e-6, out=out)
        '''
    )
    obj.run(pytorch_code, ['result', 'out'])