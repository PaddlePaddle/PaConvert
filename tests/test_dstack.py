
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap

from tests.apibase import APIBase

obj = APIBase('torch.dstack')

def _test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([4, 5, 6])
        result = torch.dstack((a, b))
        '''
    )
    obj.run(pytorch_code, ['result'])

def _test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([[1],[2],[3]])
        b = torch.tensor([[4],[5],[6]])
        result = torch.dstack((a, b))
        '''
    )
    obj.run(pytorch_code, ['result'])

def _test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([[[1, 1],[2, 2],[3, 3]]])
        b = torch.tensor([[[4],[5],[6]]])
        result = torch.dstack((a, b))
        '''
    )
    obj.run(pytorch_code, ['result'])

def _test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.dstack((torch.tensor([[[1, 1],[2, 2],[3, 3]]]), torch.tensor([[[4],[5],[6]]])))
        '''
    )
    obj.run(pytorch_code, ['result'])

def _test_case_5():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        out = torch.tensor([[[1, 1],[2, 2],[3, 3]]])
        result = torch.dstack((torch.tensor([[[1, 1],[2, 2],[3, 3]]]), torch.tensor([[[4],[5],[6]]])), out=out)
        '''
    )
    obj.run(pytorch_code, ['result', 'out'])
