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

obj = APIBase("torch.Tensor.divide")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([ 0.5950,-0.0872, 2.3298, -0.2972])
        result = a.divide(torch.tensor([0.5]))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([ 0.5950,-0.0872, 2.3298, -0.2972])
        result = a.divide(0.5)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([ 0.5950,-0.0872, 2.3298, -0.2972])
        result = a.divide(other=0.5)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[ 0.5950,-0.0872], [2.3298, -0.2972]])
        b = torch.tensor([0.1815, -1.0111])
        result = a.divide(other=b)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[ 0.5950,-0.0872], [2.3298, -0.2972]])
        b = torch.tensor([0.1815, -1.0111])
        result = a.divide(other=b, rounding_mode=None)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[ 0.5950,-0.0872], [2.3298, -0.2972]])
        b = torch.tensor([0.1815, -1.0111])
        result = a.divide(other=b, rounding_mode="trunc")
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[ 0.5950,-0.0872], [2.3298, -0.2972]])
        b = torch.tensor([0.1815, -1.0111])
        result = a.divide(other=b, rounding_mode="floor")
        """
    )
    obj.run(pytorch_code, ["result"])


# paddle not support type promote
# torch.divide(int, int) return float, but paddle return int
def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[4, 9, 8]])
        b = torch.tensor([2, 3, 4])
        result = a.divide(b)
        """
    )
    obj.run(pytorch_code, ["result"], check_dtype=False)


# paddle not support type promote
# torch.divide(int, int) return float, but paddle return int, when can not divide exactly,
# paddle result equal to trunc divide, result is wrong
def _test_case_9():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[4, 3, 8]])
        b = torch.tensor([3, 2, 5])
        result = a.divide(other=b)
        """
    )
    obj.run(pytorch_code, ["result"], check_dtype=False)


def test_case_10():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[ 0.5950,-0.0872], [2.3298, -0.2972]])
        b = torch.tensor([0.1815, -1.0111])
        result = a.divide(rounding_mode=None, other=b)
        """
    )
    obj.run(pytorch_code, ["result"])
