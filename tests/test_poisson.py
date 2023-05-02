
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap
from tests.apibase import APIBase

class PoissonAPI(APIBase):

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

obj = PoissonAPI('torch.poisson')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        rates = torch.rand(4, 4) * 5
        result = torch.poisson(rates)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        rates = torch.tensor([[1., 3., 4.], [2., 3., 6.]])
        result = torch.poisson(rates)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.poisson(torch.tensor([[1., 3., 4.], [2, 3, 6]]))
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.poisson(torch.tensor([[1., 3., 4.], [2, 3, 6]]), generator=torch.Generator())
        '''
    )
    obj.run(pytorch_code, ['result'])