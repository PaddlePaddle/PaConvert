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

obj = APIBase("torch.zeros_like")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.empty(2, 3)
        result = torch.zeros_like(input)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.zeros_like(torch.empty(2, 3))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.zeros_like(torch.empty(2, 3), dtype=torch.float64, requires_grad=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        flag = False
        result = torch.zeros_like(torch.empty(2, 3), dtype=torch.float64, requires_grad=flag)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.zeros_like(torch.empty(2, 3), layout=torch.strided, dtype=torch.float64, requires_grad=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.zeros_like(input=torch.empty(2, 3), dtype=torch.float64, layout=torch.strided, device=torch.device('cpu'), requires_grad=True, memory_format=torch.preserve_format)
        """
    )
    obj.run(pytorch_code, ["result"])
