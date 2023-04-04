
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap
from tests.apibase import APIBase


obj = APIBase('torch.addmv')

def _test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        M = torch.randn(2)
        mat = torch.randn(2, 3)
        vec = torch.randn(3)
        result = torch.addmv(M, mat, vec)
        '''
    )
    obj.run(pytorch_code, ['result'], __file__)

def _test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        M = torch.randn(2)
        mat = torch.randn(2, 3)
        vec = torch.randn(3)
        result = torch.addmv(M, mat, vec, beta=0.3, alpha=0.5)
        '''
    )
    obj.run(pytorch_code, ['result'], __file__)

def _test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        M = torch.randn(3)
        mat = torch.randn(3, 4)
        vec = torch.randn(4)
        out = torch.Tensor(1)
        torch.addmv(M, mat, vec, beta=0.2, alpha=0.7, out=out)
        '''
    )
    obj.run(pytorch_code, ['out'], __file__)

def _test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        M = torch.randn(3)
        mat = torch.randn(3, 4)
        vec = torch.randn(4)
        out = torch.Tensor(1)
        torch.addmv(input=M, mat=mat, vec=vec, beta=0.2, alpha=0.7, out=out)
        '''
    )
    obj.run(pytorch_code, ['out'], __file__)

def _test_case_5():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        vec = torch.randn(4)
        out = torch.Tensor(1)
        torch.addmv(torch.randn(3), torch.randn(3, 4), vec=vec, beta=0.2, alpha=0.7, out=out)
        '''
    )
    obj.run(pytorch_code, ['out'], __file__)