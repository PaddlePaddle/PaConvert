import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap
import numpy as np
from tests.apibase import APIBase


class InstanceNorm3dAPIBase(APIBase):
    def check(self, pytorch_result, paddle_result):
        if pytorch_result.requires_grad:
            torch_numpy, paddle_numpy = pytorch_result.detach().numpy(), paddle_result.numpy()
        else:
            torch_numpy, paddle_numpy = pytorch_result.numpy(), paddle_result.numpy()

        if not np.allclose(paddle_numpy, torch_numpy):
            return False
        if str(pytorch_result.dtype)[6:] != str(paddle_result.dtype)[7:]:
            return False
        return True

obj = InstanceNorm3dAPIBase('torch.nn.InstanceNorm3d')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch.nn as nn
        import torch
        m = nn.InstanceNorm3d(100)
        input = torch.ones(20, 100, 35, 45, 10)
        result = m(input)
        '''
    )
    obj.run(pytorch_code,['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch.nn as nn
        import torch
        m = nn.InstanceNorm3d(100, affine=True)
        input = torch.ones(20, 100, 35, 45, 10)
        result = m(input)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch.nn as nn
        import torch
        m = nn.InstanceNorm3d(100, affine=False)
        input = torch.ones(20, 100, 35, 45, 10)
        result = m(input)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch.nn as nn
        import torch
        m = nn.InstanceNorm3d(100, affine=True, momentum=0.1)
        input = torch.ones(20, 100, 35, 45, 10)
        result = m(input)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_5():
    pytorch_code = textwrap.dedent(
        '''
        import torch.nn as nn
        import torch
        m = nn.InstanceNorm3d(100, affine=False, momentum=0.1)
        input = torch.ones(20, 100, 35, 45, 10)
        result = m(input)
        '''
    )
    obj.run(pytorch_code, ['result'])