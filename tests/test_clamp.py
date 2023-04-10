
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap
from tests.apibase import APIBase


obj = APIBase('torch.clamp')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([-1.7120,  0.1734, -0.0478, 0.8922])
        result = torch.clamp(a, -0.5, 0.5)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([-1.7120,  0.1734, -0.0478, 0.8922])
        result = torch.clamp(a, min=-0.2, max=0.5)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([-1.7120,  0.1734, -0.0478, 0.8922])
        result = torch.clamp(a, min=-0.5, max=0.5)
        '''
    )
    obj.run(pytorch_code, ['result'])

def _test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([-1.7120,  0.1734, -0.0478, 0.8922])
        min = torch.tensor([-0.3, 0.04, 0.23, 0.98])
        result = torch.clamp(a, min=min)
        '''
    )
    obj.run(pytorch_code, ['result'])

def _test_case_5():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([-1.7120,  0.1734, -0.0478, 0.8922])
        max = torch.tensor([-0.3, 0.04, 0.23, 0.98])
        result = torch.clamp(a, max=max)
        '''
    )
    obj.run(pytorch_code, ['result'])
