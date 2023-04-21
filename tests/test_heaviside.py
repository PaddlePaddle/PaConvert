
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap

from tests.apibase import APIBase

obj = APIBase('torch.heaviside')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        input = torch.tensor([-1.5, 0, 2.0])
        values = torch.tensor([0.5])
        torch.heaviside(input, values)
        '''
    )
    obj.run(pytorch_code, ['input'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        input = torch.tensor([-1.5, 0, 2.0])
        values = torch.tensor([0.5, 1., 9])
        torch.heaviside(input, values)
        '''
    )
    obj.run(pytorch_code, ['input'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.heaviside(torch.tensor([-1.5, 0, 2.0]), torch.tensor([0.5]))
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.heaviside(input=torch.tensor([-1.5, 0, 2.0]), values=torch.tensor([0.5, 1., 9]))
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_5():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        out = torch.tensor([2., 3.])
        result = torch.heaviside(input=torch.tensor([-1.5, 0, 2.0]), values=torch.tensor([0.5, 1., 9]), out=out)
        '''
    )
    obj.run(pytorch_code, ['result', 'out'])