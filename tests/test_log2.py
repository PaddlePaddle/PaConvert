
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap
from tests.apibase import APIBase


obj = APIBase('torch.log2')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        input = torch.tensor([4.7767, 4.3234, 1.2156, 0.2411, 4.5739])
        result = torch.log2(input)
        '''
    )
    obj.run(pytorch_code,['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.log2(torch.tensor([4.7767, 4.3234, 1.2156, 0.2411, 4.5739]))
        '''
    )
    obj.run(pytorch_code,['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        input = torch.tensor([4.7767, 4.3234, 1.2156, 0.2411, 4.5739])
        out = torch.tensor([4.7767, 4.3234, 1.2156, 0.2411, 4.5739])
        result = torch.log2(input, out=out)
        '''
    )
    obj.run(pytorch_code,['result'])

def _test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.log2(torch.tensor([4, 10, 7, 9]))
        '''
    )
    obj.run(pytorch_code,['result'])