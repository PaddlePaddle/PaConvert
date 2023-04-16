
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap

from tests.apibase import APIBase

obj = APIBase('torch.gather')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([[1, 2], [3, 4]])
        result = torch.gather(a, 1, torch.tensor([[0, 0], [1, 0]]))
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.gather(torch.tensor([[1, 2], [3, 4]]), 1, torch.tensor([[0, 0], [1, 0]]))
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        dim = 0
        index = torch.tensor([[0, 0], [1, 0]])
        result = torch.gather(torch.tensor([[1, 2], [3, 4]]), dim, index)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.gather(input=torch.tensor([[1, 2], [3, 4]]), dim=1, index=torch.tensor([[0, 0], [1, 0]]))
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_5():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([[1, 2], [3, 4]])
        out = torch.tensor(a)
        result = torch.gather(a, 1, torch.tensor([[0, 0], [1, 0]]), out=out)
        '''
    )
    obj.run(pytorch_code, ['result', 'out'])
