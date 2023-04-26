
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap

from tests.apibase import APIBase

obj = APIBase('torch.unbind')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x = torch.tensor([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]])
        result = torch.unbind(x)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x = torch.tensor([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]])
        result = torch.unbind(x, 1)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x = torch.tensor([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]])
        result = torch.unbind(input=x, dim=0)
        '''
    )
    obj.run(pytorch_code, ['result'])