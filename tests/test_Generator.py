import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap
import numpy as np
from tests.apibase import APIBase
import paddle


class GeneratorAPIBase(APIBase):
    def check(self, pytorch_result, paddle_result):
        if isinstance(paddle_result, paddle.fluid.libpaddle.Generator):
            return True
        return False

obj = GeneratorAPIBase('torch.Generator')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.Generator(device='cpu')
        '''
    )
    obj.run(pytorch_code,['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.Generator()
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.Generator('cpu')
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        if torch.cuda.is_available():
            result = torch.Generator('cuda')
        else:
            result = torch.Generator('cpu')
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_5():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        if torch.cuda.is_available():
            result = torch.Generator(device='cuda')
        else:
            result = torch.Generator(device='cpu')
        '''
    )
    obj.run(pytorch_code, ['result'])