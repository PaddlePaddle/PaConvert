
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap

from tests.apibase import APIBase

obj = APIBase('torch.index_select')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x = torch.eye(2, 4)
        indices = torch.tensor([0, 1])
        result = torch.index_select(x, 0, indices)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        indices = torch.tensor([0, 1])
        result = torch.index_select(torch.eye(3, 4), 1, indices)
        '''
    )
    obj.run(pytorch_code, ['result'])
    
def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        indices = torch.tensor([0, 1])
        dim = 0
        result = torch.index_select(torch.eye(3, 4), dim, indices)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        indices = torch.tensor([0, 3])
        dim = 0
        result = torch.index_select(input=torch.eye(5, 4), dim=dim, index=indices)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_5():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        indices = torch.tensor([0, 3])
        out = torch.eye(5, 4)
        result = torch.index_select(torch.eye(5, 4), 1, indices, out=out)
        '''
    )
    obj.run(pytorch_code, ['result', 'out'])