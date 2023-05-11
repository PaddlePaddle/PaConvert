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

import os
import sys

sys.path.append(os.path.dirname(__file__) + "/../")

import textwrap

from tests.apibase import APIBase

obj = APIBase("torch.fmax")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.fmax(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[1, 2], [3, 4]])
        other = torch.tensor([[1, 1], [4, 4]])
        result = torch.fmax(input, other)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[1, 2], [3, 4]])
        other = torch.tensor([[1, 2], [3, 4]])
        result = torch.fmax(input, other)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[1, 2], [3, 4]])
        other = torch.tensor([1, 2])
        result = torch.fmax(input, other)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[1, 2], [3, 4]])
        other = torch.tensor([[1, 2], [3, 4]])
        out = torch.tensor([1, 2])
        result = torch.fmax(input, other, out=out)
        """
    )
    obj.run(pytorch_code, ["result", "out"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([9.7, float('nan'), 3.1])
        other = torch.tensor([-2.2, 0.5, float('nan')])
        result = torch.fmax(input, other)
        """
    )
    obj.run(pytorch_code, ["result"])
