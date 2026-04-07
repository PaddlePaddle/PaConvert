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

obj = APIBase("torch.signbit")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([-0., 1.1, -2.1, 0., 2.5], dtype=torch.float32)
        result = torch.signbit(x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([-0., 1.1, -2.1, 0., 2.5], dtype=torch.float32)
        result = torch.signbit(input=x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([-0., 1.1, -2.1, 0., 2.5], dtype=torch.float32)
        out = torch.tensor([], dtype=torch.bool)
        result = torch.signbit(out=out, input=x)
        """
    )
    obj.run(pytorch_code, ["result", "out"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([-0., 1.1, -2.1, 0., 2.5], dtype=torch.float32)
        out = torch.tensor([], dtype=torch.bool)
        result = torch.signbit(input=x, out=out)
        """
    )
    obj.run(pytorch_code, ["result", "out"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([-0., 1.1, -2.1, 0., 2.5], dtype=torch.float32)
        out = torch.tensor([], dtype=torch.bool)
        result = torch.signbit(x, out=out)
        """
    )
    obj.run(pytorch_code, ["result", "out"])


def test_case_6():
    """Test with float64 dtype"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([-0., 1.1, -2.1, 0., 2.5], dtype=torch.float64)
        result = torch.signbit(x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    """Test with 2D tensor"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[-1., 2., -3.], [4., -5., 6.]], dtype=torch.float32)
        result = torch.signbit(input=x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    """Test with 3D tensor and out parameter"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[[-0., 1.1], [-2.1, 0.]], [[2.5, -3.], [4., -5.]]], dtype=torch.float32)
        out = torch.zeros(2, 2, 2, dtype=torch.bool)
        torch.signbit(x, out=out)
        """
    )
    obj.run(pytorch_code, ["out"])


def test_case_9():
    """Test with all positive values"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1., 2., 3., 0.5, 10.], dtype=torch.float32)
        result = torch.signbit(input=x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_10():
    """Test with all negative values"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([-1., -2., -3., -0.5, -10.], dtype=torch.float32)
        result = torch.signbit(x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_11():
    """Test with single element tensor"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([-5.], dtype=torch.float32)
        result = torch.signbit(input=x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_12():
    """Test with keyword arguments out of order"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([-0., 1.1, -2.1, 0., 2.5], dtype=torch.float32)
        out = torch.zeros(5, dtype=torch.bool)
        result = torch.signbit(out=out, input=x)
        """
    )
    obj.run(pytorch_code, ["result", "out"])


def test_case_13():
    """Test with out parameter pre-allocated"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1., -2., 3., -0.], dtype=torch.float32)
        out = torch.zeros(4, dtype=torch.bool)
        result = torch.signbit(input=x, out=out)
        result_is_same = result is out
        """
    )
    obj.run(pytorch_code, ["out", "result_is_same"], check_value=False)
