# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

obj = APIBase("torch.nn.modules.conv.Conv2d")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        model = torch.nn.modules.conv.Conv2d(3, 6, 3)
        result = model(torch.randn(1, 3, 8, 8))
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        model = torch.nn.modules.conv.Conv2d(in_channels=3, out_channels=6, kernel_size=3)
        result = model(torch.randn(1, 3, 8, 8))
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        model = torch.nn.modules.conv.Conv2d(3, 6, 3, stride=2)
        result = model(torch.randn(1, 3, 8, 8))
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        model = torch.nn.modules.conv.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=2, padding=1)
        result = model(torch.randn(1, 3, 8, 8))
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        model = torch.nn.modules.conv.Conv2d(3, 6, 3, padding=1)
        result = model(torch.randn(1, 3, 8, 8))
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        model = torch.nn.modules.conv.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        result = model(torch.randn(1, 3, 8, 8))
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        model = torch.nn.modules.conv.Conv2d(3, 12, 3, stride=2, padding=1)
        result = model(torch.randn(1, 3, 8, 8))
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        model = torch.nn.modules.conv.Conv2d(3, 6, (3, 5))
        result = model(torch.randn(1, 3, 8, 8))
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_9():
    pytorch_code = textwrap.dedent(
        """
        import torch
        model = torch.nn.modules.conv.Conv2d(3, 6, 3, bias=False)
        result = model(torch.randn(1, 3, 8, 8))
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_10():
    pytorch_code = textwrap.dedent(
        """
        import torch
        model = torch.nn.modules.conv.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=2, padding=1, dilation=2)
        result = model(torch.randn(1, 3, 8, 8))
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)
