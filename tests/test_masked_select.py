
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap

from tests.apibase import APIBase

obj = APIBase('torch.masked_select')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x = torch.eye(2, 4)
        mask = x > 0
        result = torch.masked_select(x, mask)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x = torch.ones(2, 4)
        result = torch.masked_select(x, x>0)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x = torch.ones(2, 4)
        out = torch.ones(2, 4)
        result = torch.masked_select(x, mask=x>0, out=out)
        '''
    )
    obj.run(pytorch_code, ['result', 'out'])

def _test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x = torch.eye(2, 4)
        mask = torch.tensor([True, True, True, True])
        result = torch.masked_select(x, mask)
        '''
    )
    obj.run(pytorch_code, ['result'])
