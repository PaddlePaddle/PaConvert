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
    """1D tensor input"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([-1.5, -0.5, 0.0, 0.5, 1.5])
        result = torch.relu(x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    """2D tensor input"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[-1.0, 2.0], [3.0, -4.0]])
        result = torch.relu(x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    """float64 dtype"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([-1.5, -0.5, 0.0, 0.5, 1.5], dtype=torch.float64)
        result = torch.relu(x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    """4D tensor (batch, channel, height, width)"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[[[-1.0, 2.0], [3.0, -4.0]], [[0.5, -0.5], [-1.5, 1.5]]]])
        result = torch.relu(x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    """All negative values"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([-1.0, -2.0, -3.0, -4.0])
        result = torch.relu(x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    """All positive values"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = torch.relu(x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    """Gradient computation"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([-1.0, 0.0, 1.0, 2.0], requires_grad=True)
        result = torch.relu(x)
        result.sum().backward()
        x_grad = x.grad
        """
    )
    obj.run(pytorch_code, ["result", "x_grad"], check_stop_gradient=False)


def test_case_10():
    """3D tensor"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[[-1.0, 0.5, 1.0], [2.0, -0.5, -1.5]]])
        result = torch.relu(x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_11():
    """Expression as input"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1.0, 2.0, 3.0])
        result = torch.relu(x - 2.0)
        """
    )
    obj.run(pytorch_code, ["result"])
