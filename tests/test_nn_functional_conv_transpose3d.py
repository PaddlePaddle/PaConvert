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

obj = APIBase("torch.nn.functional.conv_transpose3d")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.randn(20, 16, 10, 10, 10)
        weight = torch.randn(16, 33, 2, 2, 2)
        result = F.conv_transpose3d(x, weight)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.randn(20, 16, 10, 10, 10)
        weight = torch.randn(16, 33, 2, 2, 2)
        bias = torch.randn(33)
        result = F.conv_transpose3d(x, weight, bias)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.randn(20, 16, 10, 10, 10)
        weight = torch.randn(16, 33, 2, 2, 2)
        result = F.conv_transpose3d(x, weight, stride=2)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.randn(20, 16, 10, 10, 10)
        weight = torch.randn(16, 33, 2, 2, 2)
        result = F.conv_transpose3d(x, weight, stride=2, padding=2)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.randn(20, 16, 10, 10, 10)
        weight = torch.randn(16, 33, 2, 2, 2)
        result = F.conv_transpose3d(x, weight, stride=2, padding=2, dilation=1)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.randn(20, 16, 10, 10, 10)
        weight = torch.randn(16, 8, 2, 2, 2)
        result = F.conv_transpose3d(x, weight, stride=2, padding=2, dilation=1, groups=2)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.randn(20, 16, 10, 10, 10)
        weight = torch.randn(16, 8, 2, 2, 2)
        result = F.conv_transpose3d(x, weight, stride=2, padding=2, output_padding=1, dilation=1, groups=2)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)
