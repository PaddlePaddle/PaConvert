
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap
from tests.apibase import APIBase


obj = APIBase('torch.fix')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        input = torch.tensor([3.4742,  0.5466, -0.8008, -0.9079])
        result = torch.fix(input)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.fix(torch.tensor([3.4742,  0.5466, -0.8008, -0.9079]))
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.fix(input=torch.tensor([3.4742,  0.5466, -0.8008, -0.9079]))
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        input = torch.tensor([3.4742,  0.5466, -0.8008, -0.9079])
        out = torch.tensor([3.4742,  0.5466, -0.8008, -0.9079])
        result = torch.fix(input, out=out)
        '''
    )
    obj.run(pytorch_code, ['result', 'out'])

def test_case_5():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.fix(torch.tensor([3,  0, 5, -9]))
        '''
    )
    obj.run(pytorch_code, ['result'])