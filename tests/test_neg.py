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

obj = APIBase("torch.neg")


def test_case_1():
    """Basic usage with positional argument"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.neg(torch.tensor([1.0, -2.0, 3.0]))
    """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    """With keyword argument (input)"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1.0, -2.0, 3.0])
        result = torch.neg(input=x)
    """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    """Out parameter with positional and keyword arguments"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1.0, -2.0, 3.0])
        out = torch.empty_like(x)
        result = torch.neg(x, out=out)
    """
    )
    obj.run(pytorch_code, ["out"])


def test_case_4():
    """Out parameter with all keyword arguments"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1.0, -2.0, 3.0])
        out = torch.empty_like(x)
        result = torch.neg(input=x, out=out)
    """
    )
    obj.run(pytorch_code, ["out"])


def test_case_5():
    """Out parameter keyword in different order"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1.0, -2.0, 3.0])
        out = torch.empty_like(x)
        result = torch.neg(out=out, input=x)
    """
    )
    obj.run(pytorch_code, ["out"])


def test_case_6():
    """2D tensor"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, -2.0], [3.0, -4.0]])
        result = torch.neg(x)
    """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    """3D tensor"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[[1.0, -2.0], [3.0, -4.0]], [[5.0, -6.0], [7.0, -8.0]]])
        result = torch.neg(x)
    """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    """Integer tensor"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1, -2, 3, -4])
        result = torch.neg(x)
    """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    """Float64 tensor"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float64)
        result = torch.neg(x)
    """
    )
    obj.run(pytorch_code, ["result"])


def test_case_10():
    """Expression argument"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.neg(torch.tensor([1.0, -2.0, 3.0]) + torch.tensor([0.5, 0.5, 0.5]))
    """
    )
    obj.run(pytorch_code, ["result"])


def test_case_11():
    """Gradient computation"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1.0, -2.0, 3.0], requires_grad=True)
        y = torch.neg(x)
        y.sum().backward()
        x_grad = x.grad
        """
    )
    obj.run(pytorch_code, ["y", "x_grad"], check_stop_gradient=False)
