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

obj = APIBase("torch.add")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.add(torch.tensor([1, 2, 3]), torch.tensor([1, 4, 6]))
        """
    )
    obj.run(pytorch_code, ["result"])


# paddle not support input python number, x/y must be Tensor
def _test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.add(torch.tensor([1, 2, 3]), 20)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([20])
        result = torch.add(a, b)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([20])
        result = torch.add(a, b, alpha = 10)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([1, 2, 3])
        result = torch.add(a, torch.tensor([1, 4, 6]), alpha = 10, out=a)
        """
    )
    obj.run(pytorch_code, ["a"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([1, 2, 3])
        result = torch.add(input=a, other=torch.tensor([1, 4, 6]), alpha = 10, out=a)
        """
    )
    obj.run(pytorch_code, ["a"])


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([1, 2, 3])
        result = torch.add(other=torch.tensor([1, 4, 6]), out=a, input=a, alpha = 10)
        """
    )
    obj.run(pytorch_code, ["a"])


# current type promotion only support calculations between floating-point numbers and between complex and real numbers
def _test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.add(torch.tensor([1, 2, 3], dtype=torch.int32), torch.tensor([1, 4, 6], dtype=torch.float32))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.add(torch.tensor([1, 2, 3], dtype=torch.float64), torch.tensor([1, 4, 6], dtype=torch.float32))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_10():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.add(torch.tensor([1, 2, 3], dtype=torch.float16), torch.tensor([1, 4, 6], dtype=torch.float32))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_11():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.add(torch.tensor([1, 2, 3], dtype=torch.float64), torch.tensor([1, 4, 6], dtype=torch.complex64))
        """
    )
    obj.run(pytorch_code, ["result"])


# AI生成case
def _test_case_12():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.add(torch.tensor([1, 2, 3]), torch.tensor([1, 4, 6]), out=torch.empty(3))
        """
    )
    obj.run(pytorch_code, ["result"])


# AI生成case
def _test_case_13():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1., 2., 3.], requires_grad=True)
        y = torch.tensor([1., 4., 6.], requires_grad=True)
        z = torch.add(x, y)
        z.backward(torch.ones_like(z))
        """
    )
    obj.run(pytorch_code, ["z", "x.grad", "y.grad"])


# AI生成case
def test_case_14():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.add(torch.tensor([1, 2, 3]), torch.tensor([1]))
        """
    )
    obj.run(pytorch_code, ["result"])


# AI生成case
def _test_case_15():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.add(torch.tensor([1, 2, 3]), torch.tensor([1, 4, 6]), alpha=0.5)
        """
    )
    obj.run(pytorch_code, ["result"])


# AI生成case
@pytest.mark.skip(reason="add with sparse tensors not supported")
def test_case_16():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[0, 1], [2, 0]]).to_sparse()
        y = torch.tensor([[1, 0], [0, 2]]).to_sparse()
        result = torch.add(x, y)
        """
    )
    obj.run(pytorch_code, ["result"])
