
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap
from tests.apibase import APIBase

obj = APIBase('torch.Tensor.requires_grad')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        data = torch.tensor([23.,32., 43.])
        result = 1
        if not data.requires_grad:
            result = 2
        '''
    )
    obj.run_bool(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        data = torch.tensor([23.,32., 43.])
        result = data.requires_grad
        '''
    )
    obj.run_bool(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        data = torch.tensor([23.,32., 43.])
        data.requires_grad = False
        result = data.requires_grad
        '''
    )
    obj.run_bool(pytorch_code, ['result'])

def test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        data = torch.tensor([23.,32., 43.])
        data = torch.tensor([23.,32., 43.], requires_grad=data.requires_grad)
        result = data.requires_grad
        '''
    )
    obj.run_bool(pytorch_code, ['result'])

def test_case_5():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        data = torch.tensor([23.,32., 43.])
        result = data.requires_grad == False
        '''
    )
    obj.run_bool(pytorch_code, ['result'])

def test_case_6():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        data = torch.tensor([23.,32., 43.])
        result = not data.requires_grad
        '''
    )
    obj.run_bool(pytorch_code, ['result'])

def test_case_7():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        data = torch.tensor([23.,32., 43.])
        result = '{} , {}'.format("1", str(data.requires_grad))
        '''
    )
    obj.run_bool(pytorch_code, ['result'])

def test_case_8():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        data = torch.tensor([23.,32., 43.])
        def test():
            return True
        data.requires_grad = test()
        result = data.requires_grad
        '''
    )
    obj.run_bool(pytorch_code, ['result'])

def test_case_9():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        data = torch.tensor([23.,32., 43.])
        z = (True, False, True)
        a, data.requires_grad, c = z
        result = data.requires_grad
        '''
    )
    obj.run_bool(pytorch_code, ['result'])