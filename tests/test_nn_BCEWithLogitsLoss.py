
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap
from tests.apibase import APIBase


obj = APIBase('torch.nn.BCEWithLogitsLoss')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        import torch.nn as nn
        loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        input = torch.tensor([1.,0.7,0.2], requires_grad=True)
        target = torch.tensor([1.,0., 0.])
        result = loss(input, target)
        '''
    )
    obj.run(pytorch_code,['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        import torch.nn as nn
        loss = nn.BCEWithLogitsLoss(weight=torch.tensor([1.0,0.2, 0.2]), reduction='none')
        input = torch.tensor([1.,0.7,0.2], requires_grad=True)
        target = torch.tensor([1.,0., 0.])
        result = loss(input, target)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        import torch.nn as nn
        loss= nn.BCEWithLogitsLoss(pos_weight = torch.ones([3]))
        input = torch.tensor([1.,0.7,0.2], requires_grad=True)
        target = torch.tensor([1.,0., 0.])
        result = loss(input, target)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        import torch.nn as nn
        loss = nn.BCEWithLogitsLoss(size_average=True)
        input = torch.tensor([1.,0.7,0.2], requires_grad=True)
        target = torch.tensor([1.,0., 0.])
        result = loss(input, target)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_5():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        import torch.nn as nn
        loss = nn.BCEWithLogitsLoss()
        input = torch.tensor([1.,0.7,0.2], requires_grad=True)
        target = torch.tensor([1.,0., 0.])
        result = loss(input, target)
        '''
    )
    obj.run(pytorch_code, ['result'])