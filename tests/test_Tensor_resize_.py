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

obj = APIBase("torch.Tensor.resize_")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[0.3817, 0.1472],
            [0.3758, 0.9468],
            [0.7074, 0.3895]])
        result = a.resize_(1, 2)
        """
    )
    obj.run(pytorch_code, ["result", "a"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[0.3817, 0.1472],
            [0.3758, 0.9468],
            [0.7074, 0.3895]])
        result = a.resize_(1, 2, memory_format=torch.contiguous_format)
        """
    )
    obj.run(pytorch_code, ["result", "a"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[0.3817, 0.1472],
            [0.3758, 0.9468],
            [0.7074, 0.3895]])
        result = a.resize_([1, 2], memory_format=torch.contiguous_format)
        """
    )
    obj.run(pytorch_code, ["result", "a"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[0.3817, 0.1472],
            [0.3758, 0.9468],
            [0.7074, 0.3895]])
        b = torch.ones(1, 2)
        result = a.resize_(b.shape, memory_format=torch.contiguous_format)
        """
    )
    obj.run(pytorch_code, ["result", "a"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[0.3817, 0.1472],
            [0.3758, 0.9468],
            [0.7074, 0.3895]])
        b = torch.ones(1, 2)
        result = a.resize_(size=[1, 2], memory_format=torch.contiguous_format)
        """
    )
    obj.run(pytorch_code, ["result", "a"])
