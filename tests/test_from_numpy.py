
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap

from tests.apibase import APIBase

obj = APIBase('torch.from_numpy')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        import numpy
        a = numpy.array([1, 2, 3])
        result = torch.from_numpy(a)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        import numpy
        result = torch.from_numpy(numpy.array([1, 2, 3]))
        '''
    )
    obj.run(pytorch_code, ['result'])