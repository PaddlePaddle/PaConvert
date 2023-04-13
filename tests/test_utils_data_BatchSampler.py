
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap
from tests.apibase import APIBase

obj = APIBase('torch.utils.data.BatchSampler')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        from torch.utils.data import BatchSampler
        result = list(BatchSampler(range(10), batch_size=3, drop_last=True))
        '''
    )
    obj.run_bool(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        from torch.utils.data import BatchSampler
        result = list(BatchSampler(range(10), batch_size=3, drop_last=False))
        '''
    )
    obj.run_bool(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = batch_sampler_train = torch.utils.data.BatchSampler(range(10), 2, drop_last=True)
        result = list(result)
        '''
    )
    obj.run_bool(pytorch_code, ['result'])

def test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        batch_size = 4
        result = batch_sampler_train = torch.utils.data.BatchSampler(range(10), batch_size, drop_last=False)
        result = list(result)
        '''
    )
    obj.run_bool(pytorch_code, ['result'])

def test_case_5():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        batch_size = 4
        result = list(torch.utils.data.BatchSampler(sampler=range(10), batch_size=batch_size, drop_last=False))
        '''
    )
    obj.run_bool(pytorch_code, ['result'])