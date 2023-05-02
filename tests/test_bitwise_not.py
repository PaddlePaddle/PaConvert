
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap
from tests.apibase import APIBase


obj = APIBase('torch.bitwise_not')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.bitwise_not(torch.tensor([-1, -2, 3], dtype=torch.int8))
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        input = torch.tensor([-1, -2, 3], dtype=torch.int8)
        result = torch.bitwise_not(input)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        input = torch.tensor([True, False, True], dtype=torch.bool)
        result = torch.bitwise_not(input)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        input = torch.tensor([-1, -2, 3], dtype=torch.int8)
        out = torch.tensor(input)
        result = torch.bitwise_not(input, out=out)
        '''
    )
    obj.run(pytorch_code, ['result', 'out'])