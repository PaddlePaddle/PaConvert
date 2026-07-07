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

obj = APIBase("torch.relu")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                            [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
        result = torch.relu(x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                            [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
        result = torch.relu(input=x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    # integer input cast to float, exercising positional arg path
    pytorch_code = textwrap.dedent(
        """
        import torch
        raw = torch.arange(-5, 6)
        x = raw.to(torch.float32) * (2 ** (-0.5 + 0.5))
        result = torch.relu(x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    # higher-rank float64 tensor with both negative and positive values via linspace grid
    pytorch_code = textwrap.dedent(
        """
        import torch
        g = torch.linspace(-4., 4., steps=33, dtype=torch.float64).reshape(3, 11)
        result = torch.relu(g)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    # expression argument passed positionally; output compared elementwise against max(x,0)
    pytorch_code = textwrap.dedent(
        """
        import torch
        base = torch.arange(start=-12., end=13.)
        x = base * ((-1.) * (base < 0))
        r = torch.relu(x)
        expected_nonneg_count = int((r >= 0).sum())
        result = (expected_nonneg_count,)
        """
    )
    obj.run(pytorch_code, ["result"])
