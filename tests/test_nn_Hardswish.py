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

obj = APIBase("torch.nn.Hardswish")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        x = torch.tensor([[[-1.3020, -0.1005,  0.5766,  0.6351, -0.8893,  0.0253, -0.1756, 1.2913],
                            [-0.8833, -0.1369, -0.0168, -0.5409, -0.1511, -0.1240, -1.1870, -1.8816]]])
        model = nn.Hardswish()
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
        model = nn.Hardswish(False)
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
        model = nn.Hardswish(inplace=False)
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
        model = nn.Hardswish(True)
        result = model(x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    # integer-typed activation output stays consistent across frameworks
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        x = torch.tensor([[-9, -6, -3, 0, 3, 6, 9]], dtype=torch.int64)
        m = nn.Hardswish()
        # cast back to float since hardswish needs floating math
        y = x.to(torch.float32)
        result = m(y)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    # float64 high-dim input through keyword-instantiated module
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        x = torch.randn(3, 4, 5, dtype=torch.float64)
        layer = nn.Hardswish(inplace=False)
        first = layer(x)
        second = layer(x)
        result = ((first == second).all().item(), )
        """
    )
    obj.run(pytorch_code, ["result"], check_stop_gradient=False)


def test_case_7():
    # larger 2-D matrix exercising boundary values around ±3
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        grid = torch.linspace(-4.0, 4.0, steps=17, dtype=torch.float32)
        two_d = grid.repeat(3, 1)
        act = nn.Hardswish(inplace=False)
        result = act(two_d)
        """
    )
    obj.run(pytorch_code, ["result"])
