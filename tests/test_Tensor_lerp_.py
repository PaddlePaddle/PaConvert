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

obj = APIBase("torch.Tensor.lerp_")


def test_case_1():
    """positional arguments"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        start = torch.tensor([1., 2., 3., 4.])
        result = start.lerp_(torch.tensor([10., 10., 10., 10.]), 0.5)
        """
    )
    obj.run(pytorch_code, ["result", "start"])


def test_case_2():
    """keyword arguments"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        start = torch.tensor([1., 2., 3., 4.])
        result = start.lerp_(end=torch.tensor([10., 10., 10., 10.]), weight=0.5)
        """
    )
    obj.run(pytorch_code, ["result", "start"])


def test_case_3():
    """reordered kwargs"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        start = torch.tensor([1., 2., 3., 4.])
        result = start.lerp_(weight=0.5, end=torch.tensor([10., 10., 10., 10.]))
        """
    )
    obj.run(pytorch_code, ["result", "start"])


def test_case_4():
    """weight as tensor"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        start = torch.tensor([1., 2., 3., 4.])
        result = start.lerp_(torch.tensor([10., 10., 10., 10.]), torch.tensor([0.1, 0.3, 0.5, 0.7]))
        """
    )
    obj.run(pytorch_code, ["result", "start"])


def test_case_5():
    """2D tensor"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        start = torch.tensor([[1., 2.], [3., 4.]])
        result = start.lerp_(torch.tensor([[10., 10.], [10., 10.]]), 0.5)
        """
    )
    obj.run(pytorch_code, ["result", "start"])
