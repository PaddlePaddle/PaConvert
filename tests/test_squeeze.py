
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap

from tests.apibase import APIBase

obj = APIBase('torch.squeeze')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x = torch.zeros(2, 1, 2, 1, 2)
        result = torch.squeeze(x)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.squeeze(torch.zeros(2, 1, 2, 1, 2))
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x = torch.zeros(2, 1, 2, 1, 2)
        result = torch.squeeze(x, 1)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x = torch.zeros(2, 1, 2, 1, 2)
        result = torch.squeeze(input=x, dim=1)
        '''
    )
    obj.run(pytorch_code, ['result'])