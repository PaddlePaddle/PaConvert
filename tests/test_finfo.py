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

obj = APIBase("torch.finfo", is_aux_api=True)


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        bits = torch.finfo(torch.float16).bits
        min = torch.finfo(torch.float16).min
        max = torch.finfo(torch.float16).max
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
        x = torch.float32
        bits = torch.finfo(x).bits
        min = torch.finfo(x).min
        max = torch.finfo(x).max
        """
    )
    obj.run(pytorch_code, ["bits", "min", "max"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.bfloat16
        bits = torch.finfo(type=x).bits
        min = torch.finfo(type=x).min
        max = torch.finfo(type=x).max
        """
    )
    obj.run(pytorch_code, ["bits", "min", "max"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1.])
        bits = torch.finfo(x.dtype).bits
        min = torch.finfo(x.dtype).min
        max = torch.finfo(x.dtype).max
        """
    )
    obj.run(pytorch_code, ["bits", "min", "max"])
