
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap
from tests.apibase import APIBase


obj = APIBase('torch.cos')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.cos(torch.tensor([1.4309,  1.2706, -0.8562,  0.9796]))
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([ 1.4309,  1.2706, -0.8562,  0.9796])
        result = torch.cos(a)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = [ 1.4309,  1.2706, -0.8562,  0.9796]
        out = torch.tensor(a)
        result = torch.cos(torch.tensor(a), out=out)
        '''
    )
    obj.run(pytorch_code, ['result', 'out'])