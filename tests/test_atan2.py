
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap
from tests.apibase import APIBase


obj = APIBase('torch.atan2')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        input = torch.tensor([ 0.9041,  0.0196, -0.3108, -2.4423])
        other = torch.tensor([ 0.2341,  0.2539, -0.6256, -0.6448])
        result = torch.atan2(input, other)
        '''
    )
    obj.run(pytorch_code,['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.atan2(torch.tensor([ 0.9041,  0.0196, -0.3108, -2.4423]), torch.tensor([ 0.2341,  0.2539, -0.6256, -0.6448]))
        '''
    )
    obj.run(pytorch_code,['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        input = torch.tensor([ 0.9041,  0.0196, -0.3108, -2.4423])
        other = torch.tensor([ 0.2341,  0.2539, -0.6256, -0.6448])
        out = torch.tensor(input)
        result = torch.atan2(input, other, out=out)
        '''
    )
    obj.run(pytorch_code,['result', 'out'])