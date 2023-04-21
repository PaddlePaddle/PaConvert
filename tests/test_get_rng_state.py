import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')
import textwrap
from tests.apibase import APIBase


obj = APIBase('torch.get_rng_state')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        torch.get_rng_state()
        '''
    )
    obj.run(pytorch_code)