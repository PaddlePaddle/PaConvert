
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap
from tests.apibase import APIBase

class NormalAPI(APIBase):

    def __init__(self, pytorch_api) -> None:
        super().__init__(pytorch_api)

    def check(self, pytorch_result, paddle_result):
        if pytorch_result.requires_grad == paddle_result.stop_gradient:
            return False
        if str(pytorch_result.dtype)[6:] != str(paddle_result.dtype)[7:]:
            return False
        if pytorch_result.numpy().shape != paddle_result.numpy().shape:
            return False
        return True

obj = NormalAPI('torch.normal')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.normal(torch.arange(1., 11.), torch.arange(1, 11))
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.normal(mean=0.5, std=torch.arange(1., 6.))
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.normal(mean=torch.arange(1., 6.))
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.normal(2, 3, size=(1, 4))
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_5():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        out = torch.zeros(5)
        result = torch.normal(mean=torch.arange(1., 6.), out=out)
        '''
    )
    obj.run(pytorch_code, ['result', 'out'])