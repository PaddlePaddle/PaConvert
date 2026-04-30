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

obj = APIBase("torch.Tensor.sgn")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([ 0.5950,-0.0872, 2.3298, -0.2972])
        result = a.sgn()
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([0.5950 + 0.3451j,-0.0872 - 0.3451j, 2.3298 + 0.3451j, -0.2972 + 0.3451j])
        result = a.sgn()
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    """Test with 2D tensor"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[0.5950, -0.0872], [2.3298, -0.2972]])
        result = a.sgn()
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    """Test with 3D tensor"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[[0.5950, -0.0872], [2.3298, -0.2972]]])
        result = a.sgn()
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    """Test with zero values"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([0., 1., 0., -1., 0.])
        result = a.sgn()
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    """Test with negative values only"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([-1., -2., -3., -0.5])
        result = a.sgn()
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    """Test with positive values only"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([1., 2., 3., 0.5])
        result = a.sgn()
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    """Test with single element tensor"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([3.5])
        result = a.sgn()
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    """Test with large values"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([1e10, -1e10, 1e-10, -1e-10])
        result = a.sgn()
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_10():
    """Test with float64 dtype"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([0.5950, -0.0872, 2.3298, -0.2972], dtype=torch.float64)
        result = a.sgn()
        """
    )
    obj.run(pytorch_code, ["result"])
