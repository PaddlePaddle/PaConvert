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
#

import textwrap

from apibase import APIBase

obj = APIBase("torch.multiply")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([0.2015, -0.4255, 2.6087])
        other = torch.tensor([0.2015, -0.4255, 2.6087])
        result = torch.multiply(input, other)
        """
    )
    obj.run(pytorch_code, ["result"])


# current type promotion only support calculations between floating-point numbers and between complex and real numbers
def _test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([0.2015, -0.4255,  2.6087])
        other = torch.tensor([2, 6, 4])
        result = torch.multiply(input, other)
        """
    )
    obj.run(pytorch_code, ["result"])


# current type promotion only support calculations between floating-point numbers and between complex and real numbers
def _test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([0.2015, -0.4255,  2.6087])
        result = torch.multiply(input, other=5)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([3, 6, 9])
        result = torch.multiply(input, other=5)
        """
    )
    obj.run(pytorch_code, ["result"])


# current type promotion only support calculations between floating-point numbers and between complex and real numbers
def _test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([0.2015, -0.4255,  2.6087])
        out = torch.tensor([0.2015, -0.4255,  2.6087])
        result = torch.multiply(input, other=5, out=out)
        """
    )
    obj.run(pytorch_code, ["result", "out"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([0.2015, -0.4255,  2.6087])
        out = torch.tensor([0.2015, -0.4255,  2.6087])
        result = torch.multiply(input=input, other=torch.tensor(5.), out=out)
        """
    )
    obj.run(pytorch_code, ["result", "out"])


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([0.2015, -0.4255,  2.6087])
        other = torch.tensor([0.2015, -0.4255,  2.6087])
        result = torch.multiply(other=other, input=input)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([0.2015, -0.4255,  2.6087], dtype=torch.float32)
        other = torch.tensor([2, 6, 4], dtype=torch.float64)
        result = torch.multiply(input, other)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([0.2015, -0.4255,  2.6087], dtype=torch.float16)
        other = torch.tensor([2, 6, 4], dtype=torch.float64)
        result = torch.multiply(input, other)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_10():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([0.2015, -0.4255,  2.6087], dtype=torch.float32)
        other = torch.tensor([2, 6, 4], dtype=torch.float16)
        result = torch.multiply(input, other)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_11():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([0.2015, -0.4255,  2.6087], dtype=torch.float32)
        other = torch.tensor([2, 6, 4], dtype=torch.complex128)
        result = torch.multiply(input, other)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_12():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([0.2015, -0.4255,  2.6087], dtype=torch.int32)
        other = torch.tensor([2, 6, 4], dtype=torch.complex64)
        result = torch.multiply(input, other)
        """
    )
    obj.run(pytorch_code, ["result"])
