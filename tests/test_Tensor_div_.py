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

obj = APIBase("torch.Tensor.div_")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([ 0.5950,-0.0872, 2.3298, -0.2972])
        a.div_(torch.tensor([0.5]))
        """
    )
    obj.run(pytorch_code, ["a"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[ 0.5950,-0.0872], [2.3298, -0.2972]])
        b = torch.tensor([0.1815, -1.0111])
        a.div_(other=b)
        """
    )
    obj.run(pytorch_code, ["a"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[ 0.5950,-0.0872], [2.3298, -0.2972]])
        b = torch.tensor([0.1815, -1.0111])
        a.div_(other=b, rounding_mode=None)
        """
    )
    obj.run(pytorch_code, ["a"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[ 0.5950,-0.0872], [2.3298, -0.2972]])
        b = torch.tensor([0.1815, -1.0111])
        a.div_(other=b, rounding_mode="trunc")
        """
    )
    obj.run(pytorch_code, ["a"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[ 0.5950,-0.0872], [2.3298, -0.2972]])
        b = torch.tensor([0.1815, -1.0111])
        a.div_(other=b, rounding_mode="floor")
        """
    )
    obj.run(pytorch_code, ["a"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([ 0.5950,-0.0872, 2.3298, -0.2972])
        a.div_(rounding_mode="trunc", other=torch.tensor([0.5]))
        """
    )
    obj.run(pytorch_code, ["a"])
