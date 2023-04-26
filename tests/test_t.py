
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap

from tests.apibase import APIBase

obj = APIBase('torch.t')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x = torch.zeros(2, 3)
        result = torch.t(x)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x = torch.zeros(2)
        result = torch.t(x)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.t(torch.zeros(2, 3))
        '''
    )
    obj.run(pytorch_code, ['result'])