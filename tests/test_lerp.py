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

obj = APIBase("torch.lerp")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        start = torch.tensor([1., 2., 3., 4.])
        end = torch.tensor([10., 10., 10., 10.])
        weight = torch.tensor([0.5, 1, 0.3, 0.6])
        result = torch.lerp(start, end, weight)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        weight = torch.tensor([0.5, 1, 0.3, 0.6])
        result = torch.lerp(torch.tensor([1., 2., 3., 4.]), torch.tensor([10., 10., 10., 10.]), weight)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        start = torch.tensor([1., 2., 3., 4.])
        end = torch.tensor([10., 10., 10., 10.])
        weight = torch.tensor([0.5, 1, 0.3, 0.6])
        result = torch.lerp(input=start, end=end, weight=weight)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        start = torch.tensor([1., 2., 3., 4.])
        end = torch.tensor([10., 10., 10., 10.])
        weight = torch.tensor([0.5, 1, 0.3, 0.6])
        out = torch.tensor([0.5, 1, 0.3, 0.6])
        result = torch.lerp(start, end, weight, out=out)
        """
    )
    obj.run(pytorch_code, ["result", "out"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        start = torch.tensor([1., 2., 3., 4.])
        end = torch.tensor([10., 10., 10., 10.])
        result = torch.lerp(input=start, end=end, weight=0.5)
        """
    )
    obj.run(pytorch_code, ["result"])
