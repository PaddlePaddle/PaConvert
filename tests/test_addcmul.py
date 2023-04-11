
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')
import textwrap
from tests.apibase import APIBase


obj = APIBase('torch.addcmul')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        tensor1 = torch.tensor([1., 2., 3.])
        tensor2 = torch.tensor([4., 5., 6.])
        input = torch.tensor([7., 8., 9.])
        out = torch.tensor([1., 1., 1.])
        result = torch.addcmul(input, tensor1, tensor2, out=out)
        '''
    )
    obj.run(pytorch_code, ['out'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        tensor1 = torch.tensor([1., 2., 3.])
        tensor2 = torch.tensor([4., 5., 6.])
        input = torch.tensor([7., 8., 9.])
        result = torch.addcmul(input, tensor1, tensor2)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        tensor1 = torch.tensor([1., 2., 3.])
        tensor2 = torch.tensor([4., 5., 6.])
        input = torch.tensor([7., 8., 9.])
        result = torch.addcmul(input, tensor1, tensor2, value=2)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        tensor1 = torch.tensor([1., 2., 3.])
        tensor2 = torch.tensor([4., 5., 6.])
        input = torch.tensor([7., 8., 9.])
        value = 5.0
        result = torch.addcmul(input, tensor1, tensor2=tensor2, value=value)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_5():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        tensor1 = torch.tensor([1., 2., 3.])
        tensor2 = torch.tensor([4., 5., 6.])
        input = torch.tensor([7., 8., 9.])
        value = 5
        result = torch.addcmul(input, tensor1, tensor2, value=value)
        '''
    )
    obj.run(pytorch_code, ['result'])