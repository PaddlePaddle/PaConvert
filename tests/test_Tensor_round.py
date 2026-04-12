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

obj = APIBase("torch.Tensor.round")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[ 0.9254, -0.6213]])
        result = a.round()
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[ 102003.9254, -12021.6213]])
        result = a.round(decimals=1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[ 102003.9254, -12021.6213]])
        result = a.round(decimals=-1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[ 102003.9254, -12021.6213]])
        result = a.round(decimals=3)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[ 102003.9254, -12021.6213]])
        result = a.round(decimals=-3)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    """Test with 3D tensor"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[[0.123, 0.456], [0.789, 0.111]], [[0.222, 0.333], [0.444, 0.555]]])
        result = a.round(decimals=2)
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        check_value=False,
        reason="paddle 0.555 round to 0.55, but torch is 0.56, torch use Banker's Rounding for .5",
    )


def test_case_7():
    """Test with float64"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[ 0.9254, -0.6213]], dtype=torch.float64)
        result = a.round()
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    """Test gradient computation"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([0.5, 1.5, 2.5, 3.5], requires_grad=True)
        y = a.round()
        y.sum().backward()
        a_grad = a.grad
        """
    )
    obj.run(pytorch_code, ["y", "a_grad"], check_stop_gradient=False)


def test_case_9():
    """Test round half to even"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([-0.5, 0.5, 1.5, 2.5])
        result = a.round()
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_10():
    """Test with default decimals=0"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([1.234, 5.678, 9.012])
        result = a.round()
        """
    )
    obj.run(pytorch_code, ["result"])
