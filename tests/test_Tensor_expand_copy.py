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

import pytest
from apibase import APIBase

obj = APIBase("torch.Tensor.expand_copy")


@pytest.mark.skip(reason="torch.Tensor.expand_copy requires PyTorch>=2.8")
def test_case_1():
    """basic expand: tensor.expand_copy(4, 2, 2)"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[[1], [2]]])
        result = a.expand_copy(4, 2, 2)
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skip(reason="torch.Tensor.expand_copy requires PyTorch>=2.8")
def test_case_2():
    """with -1 dims: tensor.expand_copy(4, -1, 2)"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[[1], [2]]])
        result = a.expand_copy(4, -1, 2)
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skip(reason="torch.Tensor.expand_copy requires PyTorch>=2.8")
def test_case_3():
    """1D to 3D"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([1, 2, 3])
        result = a.expand_copy(3, 3)
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skip(reason="torch.Tensor.expand_copy requires PyTorch>=2.8")
def test_case_4():
    """keyword arguments: expand_copy(size=(4, 2, 2))"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[[1], [2]]])
        result = a.expand_copy(size=(4, 2, 2))
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skip(reason="torch.Tensor.expand_copy requires PyTorch>=2.8")
def test_case_5():
    """1D to 3D with -1 dims via keyword"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([1, 2, 3])
        result = a.expand_copy(3, -1)
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skip(reason="torch.Tensor.expand_copy requires PyTorch>=2.8")
def test_case_6():
    """list argument: expand_copy([4, 2, 2])"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[[1], [2]]])
        result = a.expand_copy([4, 2, 2])
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skip(reason="torch.Tensor.expand_copy requires PyTorch>=2.8")
def test_case_7():
    """expand_copy with variable size"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([1, 2, 3])
        size = (3, 3)
        result = a.expand_copy(size)
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skip(reason="torch.Tensor.expand_copy requires PyTorch>=2.8")
def test_case_8():
    """expand_copy with unpacked variable size"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([1, 2, 3])
        size = (3, 3)
        result = a.expand_copy(*size)
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skip(reason="torch.Tensor.expand_copy requires PyTorch>=2.8")
def test_case_9():
    """expand_copy with keyword list"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[[1], [3]]])
        result = a.expand_copy(size=[4, 2, 2])
        """
    )
    obj.run(pytorch_code, ["result"])
