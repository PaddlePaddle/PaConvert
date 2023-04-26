
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap

from tests.apibase import APIBase

obj = APIBase('torch.nonzero')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.nonzero(torch.tensor([1, 1, 1, 0, 1]))
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.nonzero(torch.tensor([[0.6, 0.0, 0.0, 0.0],
                                    [0.0, 0.4, 0.0, 0.0],
                                    [0.0, 0.0, 1.2, 0.0],
                                    [0.0, 0.0, 0.0,-0.4]]))
        '''
    )
    obj.run(pytorch_code, ['result'])

def _test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x = torch.tensor([[0.6, 0.0, 0.0, 0.0],
                        [0.0, 0.4, 0.0, 0.0],
                        [0.0, 0.0, 1.2, 0.0],
                        [0.0, 0.0, 0.0,-0.4]])
        result = torch.nonzero(x, as_tuple=True)
        '''
    )
    obj.run(pytorch_code, ['result'])

def _test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x = torch.tensor([[0.6, 0.0, 0.0, 0.0],
                        [0.0, 0.4, 0.0, 0.0],
                        [0.0, 0.0, 1.2, 0.0],
                        [0.0, 0.0, 0.0,-0.4]])
        as_tuple = True
        result = torch.nonzero(x, as_tuple=as_tuple)
        '''
    )
    obj.run(pytorch_code, ['result'])