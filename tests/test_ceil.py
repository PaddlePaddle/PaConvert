
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap
from tests.apibase import APIBase


obj = APIBase('torch.ceil')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.ceil(torch.tensor([-0.6341, -1.4208, -1.0900,  0.5826]))
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        input = torch.tensor([-0.6341, -1.4208, -1.0900,  0.5826])
        result = torch.ceil(input)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        input = torch.tensor([-0.6341, -1.4208, -1.0900,  0.5826])
        out = torch.tensor([-0.6341, -1.4208, -1.0900,  0.5826])
        result = torch.ceil(input, out=out)
        '''
    )
    obj.run(pytorch_code, ['result', 'out'])