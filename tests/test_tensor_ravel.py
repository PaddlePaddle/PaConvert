
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap

from tests.apibase import APIBase

obj = APIBase('torch.ravel')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([[4, 9], [23, 2]])
        result = a.ravel()
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.tensor([[4, 9], [23, 2]]).ravel()
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([[4., 9.], [23., 2.]])
        result = a.ravel()
        '''
    )
    obj.run(pytorch_code, ['result'])