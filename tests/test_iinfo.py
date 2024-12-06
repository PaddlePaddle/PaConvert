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

obj = APIBase("torch.iinfo")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        bits = torch.iinfo(torch.int32).bits
        min = torch.iinfo(torch.int32).min
        max = torch.iinfo(torch.int32).max
        """
    )
    obj.run(pytorch_code, ["bits", "min", "max"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.int16
        bits = torch.iinfo(x).bits
        min = torch.iinfo(x).min
        max = torch.iinfo(x).max
        """
    )
    obj.run(pytorch_code, ["bits", "min", "max"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.uint8
        bits = torch.iinfo(type=x).bits
        min = torch.iinfo(type=x).min
        max = torch.iinfo(type=x).max
        """
    )
    obj.run(pytorch_code, ["bits", "min", "max"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1])
        bits = torch.iinfo(x.dtype).bits
        min = torch.iinfo(x.dtype).min
        max = torch.iinfo(x.dtype).max
        """
    )
    obj.run(pytorch_code, ["bits", "min", "max"])
