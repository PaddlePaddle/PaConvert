
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap

from tests.apibase import APIBase

obj = APIBase('torch.sort')

def _test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([[4, 9], [23, 2]])
        result = torch.sort(a)
        '''
    )
    obj.run(pytorch_code, ['result'])

def _test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([[4, 9], [23, 2]])
        result = torch.sort(a, 0)
        '''
    )
    obj.run(pytorch_code, ['result'])

def _test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([[4, 9], [23, 2]])
        result = torch.sort(input=a, dim=1, descending=True, stable=True)
        '''
    )
    obj.run(pytorch_code, ['result'])

def _test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        descending=False
        result = torch.sort(torch.tensor([[4, 9], [23, 2]]), dim=1, descending=descending, stable=True)
        '''
    )
    obj.run(pytorch_code, ['result'])

def _test_case_5():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([[4, 9], [23, 2]])
        out = torch.tensor(a)
        result = torch.sort(input=a, dim=1, descending=True, stable=True, out=out)
        '''
    )
    obj.run(pytorch_code, ['result', 'out'])