import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')
import textwrap
from tests.apibase import APIBase


obj = APIBase('torch.var_mean')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor(
            [[ 0.2035,  1.2959,  1.8101, -0.4644],
            [ 1.5027, -0.3270,  0.5905,  0.6538],
            [-1.5745,  1.3330, -0.5596, -0.6548],
            [ 0.1264, -0.5080,  1.6420,  0.1992]])
        result = torch.var_mean(a)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor(
            [[ 0.2035,  1.2959,  1.8101, -0.4644],
            [ 1.5027, -0.3270,  0.5905,  0.6538],
            [-1.5745,  1.3330, -0.5596, -0.6548],
            [ 0.1264, -0.5080,  1.6420,  0.1992]])
        var, mean = torch.var_mean(a)
        '''
    )
    obj.run(pytorch_code, ['var', 'mean'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor(
            [[ 0.2035,  1.2959,  1.8101, -0.4644],
            [ 1.5027, -0.3270,  0.5905,  0.6538],
            [-1.5745,  1.3330, -0.5596, -0.6548],
            [ 0.1264, -0.5080,  1.6420,  0.1992]])
        var, mean = torch.var_mean(a, dim=1)
        '''
    )
    obj.run(pytorch_code, ['var', 'mean'])

def test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor(
            [[ 0.2035,  1.2959,  1.8101, -0.4644],
            [ 1.5027, -0.3270,  0.5905,  0.6538],
            [-1.5745,  1.3330, -0.5596, -0.6548],
            [ 0.1264, -0.5080,  1.6420,  0.1992]])
        var, mean = torch.var_mean(a, dim=(0, 1))
        '''
    )
    obj.run(pytorch_code, ['var', 'mean'])

def test_case_5():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor(
            [[ 0.2035,  1.2959,  1.8101, -0.4644],
            [ 1.5027, -0.3270,  0.5905,  0.6538],
            [-1.5745,  1.3330, -0.5596, -0.6548],
            [ 0.1264, -0.5080,  1.6420,  0.1992]])
        var, mean = torch.var_mean(a, dim=1, keepdim=True)
        '''
    )
    obj.run(pytorch_code, ['var', 'mean'])

def test_case_6():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor(
            [[ 0.2035,  1.2959,  1.8101, -0.4644],
            [ 1.5027, -0.3270,  0.5905,  0.6538],
            [-1.5745,  1.3330, -0.5596, -0.6548],
            [ 0.1264, -0.5080,  1.6420,  0.1992]])
        var, mean = torch.var_mean(a, dim=1, correction=0, keepdim=True)
        '''
    )
    obj.run(pytorch_code, ['var', 'mean'])

def test_case_7():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor(
            [[ 0.2035,  1.2959,  1.8101, -0.4644],
            [ 1.5027, -0.3270,  0.5905,  0.6538],
            [-1.5745,  1.3330, -0.5596, -0.6548],
            [ 0.1264, -0.5080,  1.6420,  0.1992]])
        var, mean = torch.var_mean(a, dim=1, unbiased=False, keepdim=True)
        '''
    )
    obj.run(pytorch_code, ['var', 'mean'])