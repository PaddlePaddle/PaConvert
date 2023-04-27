
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap

from tests.apibase import APIBase

class NumelAPI(APIBase):

    def __init__(self, pytorch_api) -> None:
        super().__init__(pytorch_api)

    def check(self, pytorch_result, paddle_result):
        return pytorch_result == paddle_result

obj = NumelAPI('torch.numel')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.randn(1, 2, 3, 4, 5)
        result = torch.numel(a)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.zeros(4,4)
        result = torch.numel(a)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.numel(torch.zeros(4,4))
        '''
    )
    obj.run(pytorch_code, ['result'])

