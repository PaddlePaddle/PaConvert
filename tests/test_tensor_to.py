
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap
from tests.apibase import APIBase


obj = APIBase('torch.Tensor.to')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        cpu = torch.device('cpu')
        a =torch.ones(2, 3)
        c = torch.ones(2, 3, dtype= torch.float64, device=cpu)
        result = a.to(cpu, non_blocking=False, copy=False)
        '''
    )
    obj.run(pytorch_code,['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        cpu = torch.device('cpu')
        a =torch.ones(2, 3)
        result = a.to('cpu')
        '''
    )
    obj.run(pytorch_code,['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        cpu = torch.device('cpu')
        a =torch.ones(2, 3)
        result = a.to(device = cpu, dtype = torch.float64)
        '''
    )
    obj.run(pytorch_code,['result'])

def test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        cpu = torch.device('cpu')
        a =torch.ones(2, 3)
        result = a.to(torch.float64)
        '''
    )
    obj.run(pytorch_code,['result'])

def test_case_5():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a =torch.ones(2, 3)
        cpu = torch.device('cpu')
        result = a.to(dtype= torch.float64)
        '''
    )
    obj.run(pytorch_code,['result'])

def test_case_6():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        cpu = torch.device('cpu')
        a =torch.ones(2, 3)
        c = torch.ones(2, 3, dtype= torch.float64, device=cpu)
        result = a.to(c)
        '''
    )
    obj.run(pytorch_code,['result'])

def test_case_7():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        cpu = torch.device('cpu')
        a =torch.ones(2, 3)
        result = a.to(torch.half)
        '''
    )
    obj.run(pytorch_code,['result'])

def test_case_8():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        cpu = torch.device('cpu')
        a =torch.ones(2, 3)
        table =  a
        result = a.to(table.device)
        '''
    )
    obj.run(pytorch_code,['result'])

def test_case_9():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        cpu = torch.device('cpu')
        a =torch.ones(2, 3)
        result = a.to(torch.float32)
        '''
    )
    obj.run(pytorch_code,['result'])

def test_case_10():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.tensor([-1]).to(torch.bool)
        '''
    )
    obj.run(pytorch_code,['result'])

def test_case_11():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a =torch.ones(2, 3)
        dtype = torch.float32
        result = a.to(dtype=dtype)
        '''
    )
    obj.run(pytorch_code,['result'])

def test_case_12():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        cpu = torch.device('cpu')
        a =torch.ones(2, 3)
        result = a.to(torch.device('cpu'))
        '''
    )
    obj.run(pytorch_code,['result'])