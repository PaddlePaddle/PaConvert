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

obj = APIBase("torch.sgn")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        boundaries = torch.tensor([1, 3, 5, 7, 9])
        v = torch.tensor([[3, 6, 9], [3, 6, 9]])
        result = torch.bucketize(v, boundaries)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        boundaries = torch.tensor([1, 3, 5, 7, 9])
        v = torch.tensor([[3, 6, 9], [3, 6, 9]])
        result = torch.bucketize(input=v, boundaries=boundaries, right=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        boundaries = torch.tensor([1, 3, 5, 7, 9])
        v = torch.tensor([[3, 6, 9], [3, 6, 9]])
        result = torch.bucketize(input=v, boundaries=boundaries, out_int32=True, right=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        boundaries = torch.tensor([1, 3, 5, 7, 9])
        v = torch.tensor([[3, 6, 9], [3, 6, 9]])
        out = torch.tensor([[3, 6, 9], [3, 6, 9]])
        result = torch.bucketize(input=v, boundaries=boundaries, right=True, out=out)
        """
    )
    obj.run(pytorch_code, ["out"])
