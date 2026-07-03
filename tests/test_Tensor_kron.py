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

obj = APIBase("torch.Tensor.kron")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([4, 5, 6])
        result = a.kron(b)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([4, 5, 6])
        result = a.kron(other=b)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([4, 5, 6])
        result = a.kron(other=b)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[1, 2], [3, 4]])
        b = torch.tensor([[0, 5], [6, 7]])
        result = a.kron(b)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[[1, 2], [3, 4]]])
        b = torch.tensor([[[0, 5], [6, 7]]])
        result = a.kron(b)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([1, 2, 3], dtype=torch.float64)
        b = torch.tensor([4, 5, 6], dtype=torch.float64)
        result = a.kron(b)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.tensor([1, 2]).kron(torch.tensor([3, 4]))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    """Gradient computation test"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([1.0, 2.0], requires_grad=True)
        b = torch.tensor([3.0, 4.0], requires_grad=True)
        y = a.kron(b)
        y.sum().backward()
        a_grad = a.grad
        """
    )
    obj.run(pytorch_code, ["y", "a_grad"], check_stop_gradient=False)


def test_case_9():
    """Expression argument test"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.tensor([1, 2]).kron(torch.tensor([3, 4])) + torch.tensor([1, 1, 1, 1])
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_10():
    """Expression argument test"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.tensor([1.0, 2.0]).kron(torch.tensor([3.0, 4.0])) + torch.tensor([1.0, 1.0, 1.0, 1.0])
        """
    )
    obj.run(pytorch_code, ["result"])
