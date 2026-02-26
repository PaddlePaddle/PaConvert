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

obj = APIBase("torch.Tensor.hypot_")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([1., 2, 3])
        b = torch.tensor([4., 5, 6])
        result = a.hypot_(b)
        """
    )
    obj.run(pytorch_code, ["result", "a"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([1., 2, 3])
        b = torch.tensor([4., 5, 6])
        result = a.hypot_(other=b)
        """
    )
    obj.run(pytorch_code, ["result", "a"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([-1., 2, 3])
        b = torch.tensor([4., 5, 6])
        result = a.hypot_(other=b)
        """
    )
    obj.run(pytorch_code, ["result", "a"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([1., 2, 3])
        b = torch.tensor([4., 5, 6])
        result = a.hypot_(other=b+1)
        """
    )
    obj.run(pytorch_code, ["result", "a"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([1., 2, 3])
        b = torch.tensor([4., 5, 6])
        result = a.hypot_(b+1)
        """
    )
    obj.run(pytorch_code, ["result", "a"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[3., 5.], [8., 12.]])
        b = torch.tensor([4., 12.])
        result = a.hypot_(other=b)
        """
    )
    obj.run(pytorch_code, ["result", "a"])


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([3.0, 4.0, 5.0], requires_grad=True)
        b = torch.tensor([4.0, 3.0, 12.0])
        x = a + 0
        y = x.hypot_(b)
        y.sum().backward()
        a_grad = a.grad
        """
    )
    obj.run(pytorch_code, ["y", "a_grad"], check_stop_gradient=False)
