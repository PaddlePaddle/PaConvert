
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap
from tests.apibase import APIBase


obj = APIBase('torch.divide')

def _test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([4.0, 3.0])
        b = torch.tensor([2.0, 2.0])
        result = torch.floor_divide(a, b)
        '''
    )
    obj.run(pytorch_code, ['result'])

def _test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.floor_divide(torch.tensor([4.0, 3.0]), torch.tensor([2.0, 2.0]))
        '''
    )
    obj.run(pytorch_code, ['result'])

def _test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.floor_divide(input=torch.tensor([4.0, 3.0]), other=2)
        '''
    )
    obj.run(pytorch_code, ['result'])

def _test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        out = torch.tensor([4.0, 3.0])
        result = torch.floor_divide(input=torch.tensor([4.0, 3.0]), other=torch.tensor([2.0, 2.0]))
        '''
    )
    obj.run(pytorch_code, ['result', 'out'])