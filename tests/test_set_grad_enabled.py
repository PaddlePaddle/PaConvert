
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap
from tests.apibase import APIBase


obj = APIBase('torch.set_grad_enabled')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x = torch.tensor([1.], requires_grad=True)
        is_train = False
        with torch.set_grad_enabled(is_train):
            result = x * 2
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x = torch.tensor([1.], requires_grad=True)
        with torch.set_grad_enabled(False):
            result = x * 2
        '''
    )
    obj.run(pytorch_code, ['result'])

def _test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x = torch.tensor([1.], requires_grad=True)
        _ = torch.set_grad_enabled(False)
        result = x * 2
        '''
    )
    obj.run(pytorch_code, ['result'])
