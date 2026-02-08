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
#

import textwrap

from apibase import APIBase

obj = APIBase("torch.Tensor.bitwise_right_shift")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([-2, -7, 31], dtype=torch.int8)
        other = torch.tensor([1, 0, 3], dtype=torch.int8)
        result = input.bitwise_right_shift(other)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([-2, -7, 31], dtype=torch.int8)
        other = torch.tensor([1, 0, 3], dtype=torch.int8)
        result = input.bitwise_right_shift(other=other)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([-2, -7, 31], dtype=torch.int32)
        other = torch.tensor([1, 0, 3], dtype=torch.int32)
        result = input.bitwise_right_shift(other=other)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([20, 40, 80], dtype=torch.int64)
        other = torch.tensor([2, 1, 3], dtype=torch.int64)
        result = input.bitwise_right_shift(other)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.tensor([8, 16, 32], dtype=torch.int16).bitwise_right_shift(torch.tensor([1, 2, 3], dtype=torch.int16))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([100, 200, 300], dtype=torch.int32)
        other = torch.tensor([2, 3, 4], dtype=torch.int32)
        result = input.bitwise_right_shift(other=other)
        """
    )
    obj.run(pytorch_code, ["result"])
