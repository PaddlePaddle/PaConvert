
import re
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap

from tests.apibase import APIBase

obj = APIBase('torch.no_grad')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        x = torch.tensor([1.], requires_grad=True)
        with torch.no_grad():
            y = x * 2
        '''
    )
    obj.run(pytorch_code, ['y'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        @torch.no_grad()
        def doubler(x):
            return x * 2
        x = torch.tensor([1.], requires_grad=True)
        y = doubler(x)
        '''
    )
    obj.run(pytorch_code, ['y'])