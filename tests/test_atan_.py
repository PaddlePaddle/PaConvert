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

obj = APIBase("torch.atan_")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([-3.0, -0.5, 0.25, 2.5], dtype=torch.float32)
        result = torch.atan_(x)
        """
    )
    obj.run(pytorch_code, ["x", "result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[-4.0, -1.0], [0.5, 3.0]], dtype=torch.float64)
        result = torch.atan_(input=x)
        """
    )
    obj.run(pytorch_code, ["x", "result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor(
            [-6.0, -2.5, -1.0, -0.125, 0.125, 1.0, 2.5, 6.0], dtype=torch.float32
        ).reshape(2, 2, 2)
        args = (x,)
        result = torch.atan_(*args)
        """
    )
    obj.run(pytorch_code, ["x", "result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.atan_(torch.tensor([-1.5, 0.0, 1.5], dtype=torch.float64))
        """
    )
    obj.run(pytorch_code, ["result"])
