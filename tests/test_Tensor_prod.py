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

import textwrap

from apibase import APIBase

obj = APIBase("torch.Tensor.prod")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([1.4907, 1.0593, 1.5696])
        result = input.prod()
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[1.4907, 1.0593, 1.5696], [1.4907, 1.0593, 1.5696]])
        result = input.prod(1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[1.4907, 1.0593, 1.5696], [1.4907, 1.0593, 1.5696]])
        result = input.prod(1, keepdim=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[1.4907, 1.0593, 1.5696], [1.4907, 1.0593, 1.5696]])
        result = input.prod(dim=1, keepdim=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[1.4907, 1.0593, 1.5696], [1.4907, 1.0593, 1.5696]])
        result = input.prod(dim=1, keepdim=True, dtype=torch.float64)
        """
    )
    obj.run(pytorch_code, ["result"])
