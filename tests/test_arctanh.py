
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap
from tests.apibase import APIBase


obj = APIBase('torch.arctanh')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.arctanh(torch.tensor([ -0.9385, 0.2968, -0.8591, -0.1871 ]))
        '''
    )
    obj.run(pytorch_code,['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([ -0.9385, 0.2968, -0.8591, -0.1871 ])
        result = torch.arctanh(a)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = [ -0.9385, 0.2968, -0.8591, -0.1871 ]
        out = torch.tensor(a)
        result = torch.arctanh(torch.tensor(a), out=out)
        '''
    )
    obj.run(pytorch_code, ['out', 'result'])
