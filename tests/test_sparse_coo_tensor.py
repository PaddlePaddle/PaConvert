
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap

from tests.apibase import APIBase

obj = APIBase('torch.sparse_coo_tensor')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        i = torch.tensor([[0, 1, 1],
                          [2, 0, 2]])
        v = torch.tensor([3, 4, 5], dtype=torch.float32)
        result = torch.sparse_coo_tensor(i, v, [2, 4])
        result = result.to_dense()
        '''
    )
    obj.run(pytorch_code, ['result'])

def _test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        i = torch.tensor([[0, 1, 1],
                          [2, 0, 2]])
        v = torch.tensor([3, 4, 5], dtype=torch.float32)
        result = torch.sparse_coo_tensor(i, v, [2, 4], dtype=torch.float64)
        result = result.to_dense()
        '''
    )
    obj.run(pytorch_code, ['result'])

def _test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.sparse_coo_tensor(torch.empty([1, 0]), [], [1])
        '''
    )
    obj.run(pytorch_code, ['result'])

def _test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.sparse_coo_tensor(torch.empty([1, 0]), torch.empty([0, 2]), [1, 2])
        result = result.to_dense()
        '''
    )
    obj.run(pytorch_code, ['result'])
