
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap
from tests.apibase import APIBase


obj = APIBase('torch.floor')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        input = torch.tensor([-0.8166,  1.5308, -0.2530, -0.2091])
        result = torch.floor(input)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.floor(torch.tensor([-0.8166,  1.5308, -0.2530, -0.2091]))
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.floor(input=torch.tensor([-0.8166,  1.5308, -0.2530, -0.2091]))
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        input = torch.tensor([-0.8166,  1.5308, -0.2530, -0.2091])
        out = torch.tensor([-0.8166,  1.5308, -0.2530, -0.2091])
        result = torch.floor(input, out=out)
        '''
    )
    obj.run(pytorch_code, ['result', 'out'])

def _test_case_5():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.floor(torch.tensor([3,  0, 5, -9]))
        '''
    )
    obj.run(pytorch_code, ['result'])