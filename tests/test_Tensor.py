# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os
sys.path.append(os.path.dirname(__file__) + '/../')

import textwrap

from tests.apibase import APIBase


class TensorAPI(APIBase):

    def __init__(self, pytorch_api) -> None:
        super().__init__(pytorch_api)

    def check(self, pytorch_result, paddle_result):
        if pytorch_result.requires_grad == paddle_result.stop_gradient:
            return False
        if str(pytorch_result.dtype)[6:] != str(paddle_result.dtype)[7:]:
            return False
        return True

obj = TensorAPI('torch.Tensor')

def test_case_1():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.Tensor(2, 3)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_2():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        shape = [2, 3]
        result = torch.Tensor(*shape)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_3():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        dim1, dim2 = 2, 3
        result = torch.Tensor(dim1, dim2)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_4():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        def fun(x: torch.Tensor):
            return x * 2

        a = torch.Tensor(3, 4)
        result = fun(a)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_5():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.Tensor([[3, 4], [5, 8]])
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_6():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        a = torch.tensor([[3, 4], [5, 8]])
        result = torch.Tensor(a)
        '''
    )
    obj.run(pytorch_code, ['result'])

def test_case_7():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.Tensor((1, 2, 3))
        '''
    )
    obj.run(pytorch_code, ['result'])

def _test_case_8():
    pytorch_code = textwrap.dedent(
        '''
        import torch
        result = torch.Tensor()
        '''
    )
    obj.run(pytorch_code, ['result'])
