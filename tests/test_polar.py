
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')
import textwrap
from tests.apibase import APIBase


obj = APIBase('torch.polar')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        import numpy as np
        abs = torch.tensor([1, 2], dtype=torch.float64)
        angle = torch.tensor([np.pi / 2, 5 * np.pi / 4], dtype=torch.float64)
        result = torch.polar(abs, angle)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        import numpy as np
        abs = torch.tensor([1, 2], dtype=torch.float64)
        angle = torch.tensor([np.pi / 2, 5 * np.pi / 4], dtype=torch.float64)
        out = torch.tensor([1, 2], dtype=torch.complex128)
        result = torch.polar(abs, angle, out=out)
        '''
    )
    obj.run(pytorch_code, ['out'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        import numpy as np
        angle = torch.tensor([np.pi / 2, 5 * np.pi / 4], dtype=torch.float64)
        result = torch.polar(torch.tensor([1, 2], dtype=torch.float64), angle)
        '''
    )
    obj.run(pytorch_code, ['result'])
