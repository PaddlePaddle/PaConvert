
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap

from tests.apibase import APIBase

obj = APIBase('torch.as_tensor')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        import numpy
        a = numpy.array([1, 2, 3])
        result = torch.as_tensor(a)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        import numpy
        result = torch.as_tensor(numpy.array([1, 2, 3]))
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        import numpy
        result = torch.as_tensor([1, 2, 3])
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        import numpy
        a, b , c = 1, 2, 3
        result = torch.as_tensor((a, b, c))
        '''
    )
    obj.run(pytorch_code, ['result'])


def test_case_5():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        import numpy
        result = torch.as_tensor([1, 2, 3], dtype=torch.float64)
        '''
    )
    obj.run(pytorch_code, ['result'])