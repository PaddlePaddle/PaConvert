
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap
from tests.apibase import APIBase


obj = APIBase('torch.bitwise_xor')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.bitwise_xor(torch.tensor([-1, -2, 3], dtype=torch.int8), torch.tensor([1, 0, 3], dtype=torch.int8))
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        input = torch.tensor([-1, -2, 3], dtype=torch.int8)
        other = torch.tensor([1, 0, 3], dtype=torch.int8)
        result = torch.bitwise_xor(input, other)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        input = torch.tensor([True, False, True])
        other = torch.tensor([False, True, True])
        result = torch.bitwise_xor(input, other)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.bitwise_xor(torch.tensor([True, False, True]), torch.tensor([False, True, True]))
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_5():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        input = torch.tensor([True, False, True])
        other = torch.tensor([False, True, True])
        out = torch.tensor([True, False, False])
        result = torch.bitwise_xor(input, other, out=out)
        '''
    )
    obj.run(pytorch_code, ['result', 'out'])