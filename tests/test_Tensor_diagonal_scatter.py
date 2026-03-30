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

obj = APIBase("torch.Tensor.diagonal_scatter")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.arange(6.0).reshape((2, 3))
        src = torch.ones((2,))
        result = input.diagonal_scatter(src)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.arange(6.0).reshape((2, 3))
        src = torch.ones((2,))
        result = input.diagonal_scatter(src=src)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.arange(6.0).reshape((2, 3))
        src = torch.ones((2,))
        result = input.diagonal_scatter(offset=0, src=src, dim2=1, dim1=-2)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.arange(6.0).reshape((2, 3))
        src = torch.ones((2,))
        result = input.diagonal_scatter(src=src, offset=0, dim1=-2, dim2=1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.arange(6.0).reshape((2, 3))
        src = torch.ones((2,))
        result = input.diagonal_scatter(src, 0, -2, 1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    """Test with positive offset"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.arange(12.0).reshape((3, 4))
        src = torch.ones((3,))
        result = input.diagonal_scatter(src, offset=1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    """Test with negative offset"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.arange(12.0).reshape((3, 4))
        src = torch.ones((2,))
        result = input.diagonal_scatter(src, offset=-1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    """Test with int dtype"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.arange(6, dtype=torch.int32).reshape((2, 3))
        src = torch.ones((2,), dtype=torch.int32)
        result = input.diagonal_scatter(src)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    """Test with float64 dtype"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.arange(6.0, dtype=torch.float64).reshape((2, 3))
        src = torch.ones((2,), dtype=torch.float64)
        result = input.diagonal_scatter(src, offset=0, dim1=0, dim2=1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_10():
    """Test with 3D tensor"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.arange(24.0).reshape((2, 3, 4))
        src = torch.ones((3, 2))
        result = input.diagonal_scatter(src, dim1=0, dim2=2)
        """
    )
    obj.run(pytorch_code, ["result"])
