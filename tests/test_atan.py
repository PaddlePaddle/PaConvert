
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap
from tests.apibase import APIBase


obj = APIBase('torch.atan')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.atan(torch.tensor([ 0.2341,  0.2539, -0.6256, -0.6448]))
        '''
    )
    obj.run(pytorch_code,['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([ 0.2341,  0.2539, -0.6256, -0.6448])
        result = torch.atan(a)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = [ 0.2341,  0.2539, -0.6256, -0.6448]
        out = torch.tensor(a)
        result = torch.atan(torch.tensor(a), out=out)
        '''
    )
    obj.run(pytorch_code, ['out', 'result'])
