
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap

from tests.apibase import APIBase

obj = APIBase('torch.concat')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x = torch.zeros(2, 3)
        result = torch.concat((x, x, x))
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x = torch.zeros(2, 3)
        y = torch.zeros(2, 3)
        result = torch.concat((x, y), 0)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.concat((torch.zeros(2, 3), torch.zeros(2, 3)), dim=1)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x = torch.ones(2, 3)
        y = torch.ones(2, 3)
        result = torch.concat([x, y], 0)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_5():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x = torch.ones(2, 3)
        y = torch.ones(2, 3)
        out = torch.zeros(4, 3)
        result = torch.concat([x, y], dim=0, out=out)
        '''
    )
    obj.run(pytorch_code, ['result', 'out'])