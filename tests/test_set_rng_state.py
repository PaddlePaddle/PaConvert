import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')
import textwrap
from tests.apibase import APIBase


obj = APIBase('torch.set_rng_state')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        state = torch.get_rng_state()
        torch.set_rng_state(state)
        '''
    )
    obj.run(pytorch_code)