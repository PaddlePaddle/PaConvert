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

obj = APIBase("torch.torch.finfo")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        bits = torch.torch.finfo(torch.float32).bits
        min = torch.torch.finfo(torch.float32).min
        max = torch.torch.finfo(torch.float32).max
        """
    )
    obj.run(
        pytorch_code,
        ["bits", "min", "max"],
        check_value=False,
        check_stop_gradient=False,
    )


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.float64
        bits = torch.torch.finfo(x).bits
        min = torch.torch.finfo(x).min
        max = torch.torch.finfo(x).max
        """
    )
    obj.run(pytorch_code, ["bits", "min", "max"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1.0])
        bits = torch.torch.finfo(x.dtype).bits
        result = bits
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        bits = torch.torch.finfo(type=torch.float16).bits
        min = torch.torch.finfo(torch.float16).min
        max = torch.torch.finfo(torch.float16).max
        """
    )
    obj.run(
        pytorch_code,
        ["bits", "min", "max"],
        check_value=False,
        check_stop_gradient=False,
    )


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.torch.finfo(torch.float32).eps
        """
    )
    obj.run(pytorch_code, ["result"])
