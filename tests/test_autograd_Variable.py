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

import pytest
from apibase import APIBase

obj = APIBase("torch.autograd.Variable")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1.1, 2.2, 3.3])

        result = torch.autograd.Variable(x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1.1, 2.2, 3.3])

        result = torch.autograd.Variable(x, requires_grad=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1.1, 2.2, 3.3])
        result = torch.autograd.Variable(x, requires_grad=False)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1.1, 2.2, 3.3])
        result = torch.autograd.Variable(data=x, requires_grad=False)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1.1, 2.2, 3.3])
        result = torch.autograd.Variable(x, False)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1.1, 2.2, 3.3])
        result = torch.autograd.Variable(requires_grad=False, data=x)
        """
    )
    obj.run(pytorch_code, ["result"])


# AI生成case
def _test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.randn(3, 4)
        result = torch.autograd.Variable(x)
        """
    )
    obj.run(pytorch_code, ["result"])


# AI生成case
def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.zeros(2, 3)
        result = torch.autograd.Variable(x, requires_grad=True)
        """
    )
    obj.run(pytorch_code, ["result"])


# AI生成case
@pytest.mark.skip(reason="Variable with complex dtype not supported")
def test_case_9():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1+2j, 3+4j])
        result = torch.autograd.Variable(x)
        """
    )
    obj.run(pytorch_code, ["result"])


# AI生成case
def _test_case_10():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1., 2., 3.], requires_grad=True)
        v = torch.autograd.Variable(x)
        y = v * 2
        y.backward(torch.ones_like(y))
        """
    )
    obj.run(pytorch_code, ["y", "v.grad"])
