
import sys
import os

sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap
from tests.apibase import APIBase

class EmptyLikeAPI(APIBase):

    def __init__(self, pytorch_api) -> None:
        super().__init__(pytorch_api)

    def check(self, pytorch_result, paddle_result):
        if pytorch_result.requires_grad == paddle_result.stop_gradient:
            return False
        if str(pytorch_result.dtype)[6:] != str(paddle_result.dtype)[7:]:
            return False
        return True

obj = EmptyLikeAPI('torch.empty_like')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        input = torch.empty((2,3), dtype=torch.int32)
        result = torch.empty_like(input)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.empty_like(torch.empty(2, 3))
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.empty_like(torch.empty(2, 3), dtype=torch.float64, requires_grad=True)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        flag = False
        result = torch.empty_like(torch.empty(2, 3), dtype=torch.float64, requires_grad=flag)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_5():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.empty_like(torch.empty(2, 3), layout=torch.strided, dtype=torch.float64, requires_grad=True)
        '''
    )
    obj.run(pytorch_code, ['result'])
