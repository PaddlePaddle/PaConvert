
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap

from tests.apibase import APIBase

obj = APIBase('torch.angle')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.angle(torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j]))
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x = torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j])
        result = torch.angle(x) * 180 / 3.14159
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x = torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j])
        out = torch.tensor([2., 3.])
        result = torch.angle(x, out=out)
        '''
    )
    obj.run(pytorch_code, ['result', 'out'])