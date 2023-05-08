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

import os
import sys

sys.path.append(os.path.dirname(__file__) + "/../")

import textwrap

from tests.apibase import APIBase

obj = APIBase("torch.logical_xor")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.logical_xor(torch.tensor([True, False, True]), torch.tensor([True, False, False]))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([0, 1, 10, 0], dtype=torch.int8)
        b = torch.tensor([4, 0, 1, 0], dtype=torch.int8)
        result = torch.logical_xor(a, b)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([0, 1, 10, 0], dtype=torch.float32)
        b = torch.tensor([4, 0, 1, 0], dtype=torch.float32)
        result = torch.logical_xor(a, b)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([0, 1, 10, 0], dtype=torch.float32)
        b = torch.tensor([4, 0, 1, 0], dtype=torch.float32)
        result = torch.logical_xor(input=torch.tensor([0, 1, 10., 0.]), other=torch.tensor([4, 0, 10., 0.]))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([0, 1, 10, 0], dtype=torch.float32)
        b = torch.tensor([4, 0, 1, 0], dtype=torch.float32)
        out = torch.tensor([True, False, True, True])
        result = torch.logical_xor(a, b, out=out)
        """
    )
    obj.run(pytorch_code, ["result", "out"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([0, 1, 10, 0], dtype=torch.float32)
        b = torch.tensor([4, 0, 1, 0], dtype=torch.int8)
        result = torch.logical_xor(a, b)
        """
    )
    obj.run(pytorch_code, ["result"])
