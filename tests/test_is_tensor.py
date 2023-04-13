
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap

from tests.apibase import APIBase

class IsTensorAPI(APIBase):

    def __init__(self, pytorch_api) -> None:
        super().__init__(pytorch_api)

    def check(self, pytorch_result, paddle_result):        
        return pytorch_result == paddle_result

obj = IsTensorAPI('torch.is_tensor')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([[4, 9], [23, 2]])
        result = torch.is_tensor(a)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.is_tensor(torch.tensor([[4, 9], [23, 2]]))
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.is_tensor([[4, 9], [23, 2]])
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.is_tensor((1, 0, 3, 8))
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_5():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.is_tensor(3)
        '''
    )
    obj.run(pytorch_code, ['result'])