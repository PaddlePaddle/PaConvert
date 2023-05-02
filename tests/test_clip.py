
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap
from tests.apibase import APIBase


obj = APIBase('torch.clip')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x = torch.tensor([-1.7120,  0.1734, -0.0478, -0.0922])
        result = torch.clip(x, -0.5, 0.5)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x = torch.tensor([-1.7120,  0.1734, -0.0478, -0.0922])
        min, max = -0.5, 0.5
        result = torch.clip(x, min, max)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.clip(torch.tensor([-1.7120,  0.1734, -0.0478, -0.0922]), min=-0.5, max=0.5)
        '''
    )
    obj.run(pytorch_code, ['result'])

def _test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        min = torch.linspace(-1, 1, steps=4)
        result = torch.clip(torch.tensor([-1.7120,  0.1734, -0.0478, -0.0922]), min=min)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_5():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        out = torch.tensor([-1.7120,  0.1734, -0.0478, -0.0922])
        result = torch.clip(torch.tensor([-1.7120,  0.1734, -0.0478, -0.0922]), min=-0.5, max=0.5, out=out)
        '''
    )
    obj.run(pytorch_code, ['result', 'out'])