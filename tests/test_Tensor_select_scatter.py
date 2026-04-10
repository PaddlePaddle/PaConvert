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

obj = APIBase("torch.Tensor.select_scatter")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.zeros(2, 2)
        src = torch.ones(2)
        result = input.select_scatter(src, 0, 0)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.zeros(2, 2)
        src = torch.ones(2)
        result = input.select_scatter(src, dim=0, index=0)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.zeros(2, 2)
        src = torch.ones(2)
        result = input.select_scatter(src, dim=1, index=1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.zeros(2, 2)
        src = torch.ones(2)
        result = input.select_scatter(src=src, dim=1, index=1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.zeros(2, 2)
        src = torch.ones(2)
        result = input.select_scatter(index=1, src=src, dim=1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    """Test with 3D tensor"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.zeros((2, 3, 4)).type(torch.float32)
        src = torch.ones((2, 4)).type(torch.float32)
        result = input.select_scatter(src, 1, 1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    """Test with 3D tensor and keyword arguments"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.zeros((2, 3, 4)).type(torch.float32)
        src = torch.ones((2, 4)).type(torch.float32)
        result = input.select_scatter(src=src, dim=1, index=2)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    """Test with 4D tensor"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.zeros((2, 3, 4, 5)).type(torch.float32)
        src = torch.ones((2, 3, 5)).type(torch.float32)
        result = input.select_scatter(src, dim=2, index=1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    """Test with index=0"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.zeros((3, 4)).type(torch.float32)
        src = torch.ones(4).type(torch.float32)
        result = input.select_scatter(src, dim=0, index=0)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_10():
    """Test with float64 dtype"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.zeros((2, 3)).type(torch.float64)
        src = torch.ones(3).type(torch.float64)
        result = input.select_scatter(src, dim=0, index=1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_11():
    """Test with keyword arguments out of order"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.zeros((2, 3, 4))
        src = torch.ones((2, 4))
        result = input.select_scatter(index=1, src=src, dim=1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_12():
    """Test on last dimension"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.zeros((2, 3, 4))
        src = torch.ones((2, 3))
        result = input.select_scatter(src=src, dim=2, index=0)
        """
    )
    obj.run(pytorch_code, ["result"])
