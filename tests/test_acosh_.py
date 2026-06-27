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

obj = APIBase("torch.acosh_")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1.1, 1.5, 2.0, 3.25], dtype=torch.float32)
        result = torch.acosh_(x)
        """
    )
    obj.run(pytorch_code, ["x", "result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.01, 1.25], [2.5, 4.0]], dtype=torch.float64)
        result = torch.acosh_(input=x)
        """
    )
    obj.run(pytorch_code, ["x", "result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor(
            [1.001, 1.1, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0], dtype=torch.float32
        ).reshape(2, 2, 2)
        args = (x,)
        result = torch.acosh_(*args)
        """
    )
    obj.run(pytorch_code, ["x", "result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.acosh_(torch.tensor([1.05, 1.75, 3.5], dtype=torch.float64))
        """
    )
    obj.run(pytorch_code, ["result"])
