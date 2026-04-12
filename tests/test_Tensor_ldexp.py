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

obj = APIBase("torch.Tensor.ldexp")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([1., 2., -3., -4., 5.])
        b = torch.tensor([1., 2., -3., -4., 5.])
        result = a.ldexp(b)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.tensor([1., 2., -3., -4., 5.]).ldexp(other=torch.tensor([1., 2., -3., -4., 5.]))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([1., 2., -3., -4., 5.])
        result = a.ldexp(a) * 2
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.tensor([1.]).ldexp(torch.tensor([1., 2., -3., -4., 5.]))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    """Test with int second argument"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([1., 2., 3.])
        b = torch.tensor([1, 2, 3])
        result = a.ldexp(b)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    """Test with float64 dtype"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([1.0, 2.0], dtype=torch.float64)
        b = torch.tensor([1, 2], dtype=torch.int64)
        result = a.ldexp(b)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    """Test with negative exponent"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([1., 2., 4.])
        b = torch.tensor([-1, -2, -3])
        result = a.ldexp(b)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    """Test with zero exponent"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([1., 2., 3.])
        b = torch.tensor([0, 0, 0])
        result = a.ldexp(b)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    """Test with 2D tensors"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[1., 2.], [3., 4.]])
        b = torch.tensor([[1, 2], [3, 4]])
        result = a.ldexp(b)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_10():
    """Test with keyword argument"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([1., 2., 3.])
        result = a.ldexp(other=torch.tensor([1, 2, 3]))
        """
    )
    obj.run(pytorch_code, ["result"])
