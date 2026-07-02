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

obj = APIBase("torch.nn.modules.AvgPool1d")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        model = torch.nn.modules.AvgPool1d(3)
        result = model(torch.randn(1, 3, 8))
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        model = torch.nn.modules.AvgPool1d(kernel_size=3)
        result = model(torch.randn(1, 3, 8))
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        model = torch.nn.modules.AvgPool1d(3, stride=2)
        result = model(torch.randn(1, 3, 8))
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        model = torch.nn.modules.AvgPool1d(kernel_size=3, stride=2, padding=1)
        result = model(torch.randn(1, 3, 8))
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        model = torch.nn.modules.AvgPool1d(3, stride=2, padding=1, ceil_mode=True)
        result = model(torch.randn(1, 3, 8))
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        model = torch.nn.modules.AvgPool1d(3, stride=2, count_include_pad=False)
        result = model(torch.randn(1, 3, 8))
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        model = torch.nn.modules.AvgPool1d(kernel_size=3, stride=1, padding=1)
        result = model(torch.randn(1, 3, 8))
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_9():
    pytorch_code = textwrap.dedent(
        """
        import torch
        model = torch.nn.modules.AvgPool1d(kernel_size=5, stride=3, padding=1)
        result = model(torch.randn(1, 3, 12))
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_10():
    pytorch_code = textwrap.dedent(
        """
        import torch
        model = torch.nn.modules.AvgPool1d(kernel_size=2, stride=2)
        result = model(torch.randn(1, 3, 10))
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_11():
    """Mixed arguments test"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        model = torch.nn.modules.AvgPool1d(3, stride=2, padding=1)
        result = model(torch.randn(1, 3, 8))
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_12():
    """Keyword arguments out of order test"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        model = torch.nn.modules.AvgPool1d(padding=1, kernel_size=3, stride=2)
        result = model(torch.randn(1, 3, 8))
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)
