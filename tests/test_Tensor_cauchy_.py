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

obj = APIBase("torch.Tensor.cauchy_")


def test_case_1():
    """Default parameters"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.randn([3, 4])
        result = x.cauchy_()
    """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_2():
    """Positional arguments - Paddle style (loc, scale)"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.randn([3, 4])
        result = x.cauchy_(1.0, 2.0)
    """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_3():
    """Keyword arguments - PyTorch style (median, sigma)"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.randn([3, 4])
        result = x.cauchy_(median=1.0, sigma=2.0)
    """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_4():
    """Keyword arguments - PyTorch style (median, sigma)"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.randn([3, 4])
        result = x.cauchy_(median=1.0, sigma=2.0)
    """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_5():
    """Mixed arguments - first positional, second keyword"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.randn([3, 4])
        result = x.cauchy_(1.0, sigma=2.0)
    """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_6():
    """Mixed arguments - PyTorch style keyword"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.randn([3, 4])
        result = x.cauchy_(1.0, sigma=2.0)
    """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_7():
    """Out-of-order keyword arguments - PyTorch style"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.randn([3, 4])
        result = x.cauchy_(sigma=2.0, median=1.0)
    """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_8():
    """Out-of-order keyword arguments - PyTorch style"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.randn([3, 4])
        result = x.cauchy_(sigma=2.0, median=1.0)
    """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_9():
    """1D tensor"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.randn([10])
        result = x.cauchy_(0.5, 1.5)
    """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_10():
    """2D tensor"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.randn([5, 6])
        result = x.cauchy_(median=0.0, sigma=1.0)
    """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_11():
    """3D tensor"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.randn([2, 3, 4])
        result = x.cauchy_(median=-1.0, sigma=0.5)
    """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_12():
    """Float32 dtype"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.randn([3, 4], dtype=torch.float32)
        result = x.cauchy_(1.0, 2.0)
    """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_13():
    """Float64 dtype"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.randn([3, 4], dtype=torch.float64)
        result = x.cauchy_(median=0.0, sigma=1.0)
    """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_14():
    """Only median parameter"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.randn([3, 4])
        result = x.cauchy_(median=2.0)
    """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_15():
    """Only median parameter - PyTorch style"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.randn([3, 4])
        result = x.cauchy_(median=2.0)
    """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_16():
    """Negative loc value"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.randn([3, 4])
        result = x.cauchy_(-5.0, 3.0)
    """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_17():
    """Large scale value"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.randn([3, 4])
        result = x.cauchy_(median=0.0, sigma=10.0)
    """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_18():
    """Small scale value"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.randn([3, 4])
        result = x.cauchy_(median=0.0, sigma=0.1)
    """
    )
    obj.run(pytorch_code, ["result"], check_value=False)
