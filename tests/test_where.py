
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap

from tests.apibase import APIBase

obj = APIBase('torch.where')

# The type of data we are trying to retrieve does not match the type of data currently contained in the container
def _test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x = torch.tensor([[3, 0], [4, 9]])
        y = torch.ones(2, 2)
        result = torch.where(x>0, x, y)
        '''
    )
    obj.run(pytorch_code, ['result'])

def _test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x = torch.tensor([[3, 0], [4, 9]])
        result = torch.where(x>0, x, 90)
        '''
    )
    obj.run(pytorch_code, ['result'])

def _test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x = torch.tensor([[3, 0], [4, 9]])
        result = torch.where(x)
        '''
    )
    obj.run(pytorch_code, ['result'])
