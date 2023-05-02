
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap
from tests.apibase import APIBase


obj = APIBase('torch.asinh')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.asinh(torch.tensor([ 0.1606, -1.4267, -1.0899, -1.0250 ]))
        '''
    )
    obj.run(pytorch_code,['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([ 0.1606, -1.4267, -1.0899, -1.0250 ])
        result = torch.asinh(a)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = [ 0.1606, -1.4267, -1.0899, -1.0250 ]
        out = torch.tensor(a)
        result = torch.asinh(torch.tensor(a), out=out)
        '''
    )
    obj.run(pytorch_code, ['out', 'result'])
