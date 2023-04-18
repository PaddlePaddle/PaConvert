import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')
import textwrap
from tests.apibase import APIBase


obj = APIBase('torch.logdet')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([[[ 0.9254, -0.6213],
            [-0.5787,  1.6843]],

            [[ 0.3242, -0.9665],
            [ 0.4539, -0.0887]],

            [[ 1.1336, -0.4025],
            [-0.7089,  0.9032]]])
        result = torch.logdet(input=a)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([[[ 0.9254, -0.6213],
            [-0.5787,  1.6843]],

            [[ 0.3242, -0.9665],
            [ 0.4539, -0.0887]],

            [[ 1.1336, -0.4025],
            [-0.7089,  0.9032]]])
        result = torch.logdet(a)
        '''
    )
    obj.run(pytorch_code, ['result'])