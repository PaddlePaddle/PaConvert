# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

obj = APIBase("torch.reciprocal_")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([0.5, -2.0, 4.0, -0.25], dtype=torch.float32)
        result = torch.reciprocal_(x)
        """
    )
    obj.run(pytorch_code, ["x", "result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[0.125, -1.5], [3.0, -8.0]], dtype=torch.float64)
        result = torch.reciprocal_(input=x)
        """
    )
    obj.run(pytorch_code, ["x", "result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor(
            [0.2, -0.4, 0.8, -1.6, 2.0, -2.5, 4.0, -5.0], dtype=torch.float32
        ).reshape(2, 2, 2)
        args = (x,)
        result = torch.reciprocal_(*args)
        """
    )
    obj.run(pytorch_code, ["x", "result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.reciprocal_(
            torch.tensor([0.75, -1.25, 2.5], dtype=torch.float64)
        )
        """
    )
    obj.run(pytorch_code, ["result"])
