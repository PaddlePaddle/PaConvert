import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')
import textwrap
from tests.apibase import APIBase


obj = APIBase('torch.seed')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.seed()
        '''
    )
    obj.run(pytorch_code)

