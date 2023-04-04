
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap
from tests.apibase import APIBase


obj = APIBase('torch.addr')

def _test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        vec1 = torch.arange(1., 4.)
        vec2 = torch.arange(1., 3.)
        M = torch.zeros(3, 2)
        result = torch.addr(M, vec1, vec2)
        '''
    )
    obj.run(pytorch_code, ['result'], __file__)

def _test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        vec1 = torch.arange(1., 4.)
        vec2 = torch.arange(1., 3.)
        M = torch.zeros(3, 2)
        result = torch.addr(M, vec1, vec2, beta=0.5, alpha=0.9)
        '''
    )
    obj.run(pytorch_code, ['result'], __file__)

def _test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        vec1 = torch.arange(1, 4)
        vec2 = torch.arange(1, 3)
        M = torch.zeros(3, 2)
        out = torch.Tensor(1)
        torch.addr(M, vec1, vec2, beta=0.2, out=out)
        '''
    )
    obj.run(pytorch_code, ['out'], __file__)

def _test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        vec1 = torch.arange(1, 4)
        vec2 = torch.arange(1, 3)
        M = torch.zeros(3, 2)
        out = torch.Tensor(1)
        torch.addr(input=M, vec1=vec1, vec2=vec2, beta=0.2, alpha=0.1, out=out)
        '''
    )
    obj.run(pytorch_code, ['out'], __file__)

def _test_case_5():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.addr(torch.zeros(3, 2), torch.arange(1, 4), vec2=torch.arange(1, 3), beta=0.2, alpha=0.1)
        '''
    )
    obj.run(pytorch_code, ['result'], __file__)