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

obj = APIBase("torch.rad2deg")


def test_case_1():
    """Basic usage with positional argument"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([3.142, -3.142, 6.283, -6.283])
        result = torch.rad2deg(x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    """2D tensor"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[3.142, -3.142], [6.283, -6.283], [1.570, -1.570]])
        result = torch.rad2deg(x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    """Keyword argument"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([3.142, -3.142, 6.283, -6.283])
        result = torch.rad2deg(input=x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    """Keyword argument out of order"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([3.142, -3.142, 6.283, -6.283])
        result = torch.rad2deg(input=x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    """Gradient computation"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([3.142, -3.142], requires_grad=True)
        y = torch.rad2deg(x)
        y.sum().backward()
        x_grad = x.grad
        """
    )
    obj.run(pytorch_code, ["y", "x_grad"], check_stop_gradient=False)


def test_case_6():
    """Edge case with 3D tensor"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[[1.57, -1.57], [3.14, -3.14]]])
        result = torch.rad2deg(x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    """Expression argument"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.rad2deg(torch.tensor([1.57, 3.14]) * 1.0)
        """
    )
    obj.run(pytorch_code, ["result"])
