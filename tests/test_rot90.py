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

obj = APIBase("torch.rot90")


def test_case_1():
    """Basic usage with default k=1"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = torch.rot90(x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    """Positional argument k=2"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = torch.rot90(x, 2)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    """3D tensor with dims parameter"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
        result = torch.rot90(x, dims=[1, 2])
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    """All keyword arguments"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = torch.rot90(input=x, k=2)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    """Keyword arguments with both k and dims"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
        result = torch.rot90(input=x, k=1, dims=[1, 2])
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    """Keyword arguments out of order"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = torch.rot90(dims=[0, 1], k=2, input=x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    """Expression argument"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.rot90(torch.arange(9).reshape(3, 3), k=1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    """Gradient computation"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
        y = torch.rot90(x, k=1)
        y.sum().backward()
        x_grad = x.grad
        """
    )
    obj.run(pytorch_code, ["y", "x_grad"], check_stop_gradient=False)


def test_case_9():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.rot90(dims=[0, 1], k=1, input=torch.tensor([[1, 2, 3], [4, 5, 6]]))
        """
    )
    obj.run(pytorch_code, ["result"])
