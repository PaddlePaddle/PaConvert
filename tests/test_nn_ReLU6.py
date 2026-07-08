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

obj = APIBase("torch.nn.ReLU6")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                            [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
        model = nn.ReLU6()
        result = model(x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                            [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
        model = nn.ReLU6(False)
        result = model(x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                            [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
        model = nn.ReLU6(inplace=False)
        result = model(x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                            [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
        model = nn.ReLU6(True)
        result = model(x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        x = torch.linspace(-5., 12., steps=18, dtype=torch.float64).reshape(3, 6)
        layer = nn.ReLU6(inplace=False)
        result = layer(x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    # int32 input cast to float then activated; verify idempotent forward passes
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        raw = torch.arange(-10, 10, dtype=torch.int32).to(torch.float32)
        m = nn.ReLU6()
        a = m(raw); b = m(raw)
        result = ((a == b).all().item(), (a >= 0).all().item(), (a <= 6).all().item())
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    # higher-rank tensor with explicit inplace keyword argument
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        t = torch.arange(-12., 12., dtype=torch.float32).reshape(2, 3, 4)
        act = nn.ReLU6(inplace=False)
        result = act(t)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    # float64 input, inplace path, and output tensor aliasing
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        x = torch.tensor([-3.0, -0.5, 0.0, 2.0, 6.0, 7.5], dtype=torch.float64)
        y = x.clone()
        model = nn.ReLU6(inplace=True)
        result = model(y)
        output = y
        """
    )
    obj.run(pytorch_code, ["result", "output"])
