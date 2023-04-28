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

import os
import sys

sys.path.append(os.path.dirname(__file__) + "/../")

import textwrap

from tests.apibase import APIBase


class EmptyLikeAPI(APIBase):
    def __init__(self, pytorch_api) -> None:
        super().__init__(pytorch_api)

    def check(self, pytorch_result, paddle_result):
        if pytorch_result.requires_grad == paddle_result.stop_gradient:
            return False
        if str(pytorch_result.dtype)[6:] != str(paddle_result.dtype)[7:]:
            return False
        return True


obj = EmptyLikeAPI("torch.empty_like")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.empty((2,3), dtype=torch.int32)
        result = torch.empty_like(input)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.empty_like(torch.empty(2, 3))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.empty_like(torch.empty(2, 3), dtype=torch.float64, requires_grad=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        flag = False
        result = torch.empty_like(torch.empty(2, 3), dtype=torch.float64, requires_grad=flag)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.empty_like(torch.empty(2, 3), layout=torch.strided, dtype=torch.float64, requires_grad=True)
        """
    )
    obj.run(pytorch_code, ["result"])
