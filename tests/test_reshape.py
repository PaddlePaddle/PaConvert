
import re
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap

from tests.apibase import APIBase

obj = APIBase('torch.reshape')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.arange(4.)
        result = torch.reshape(a, (2, 2))
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.arange(9)
        shape = (3, 3)
        result = torch.reshape(a, shape)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.arange(4.)
        k = 2
        result = torch.reshape(a, (k, k))
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.arange(24.)
        result = torch.reshape(a, (-1, ))
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_5():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.reshape(input=torch.arange(4.), shape=(2, 2))
        '''
    )
    obj.run(pytorch_code, ['result'])