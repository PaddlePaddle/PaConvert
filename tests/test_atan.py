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

obj = APIBase("torch.atan")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.atan(torch.tensor([0.2341, 0.2539, -0.6256, -0.6448]))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([0.2341, 0.2539, -0.6256, -0.6448])
        result = torch.atan(a)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = [0.2341, 0.2539, -0.6256, -0.6448]
        out = torch.tensor(a)
        result = torch.atan(torch.tensor(a), out=out)
        """
    )
    obj.run(pytorch_code, ["out", "result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = [0.2341, 0.2539, -0.6256, -0.6448]
        out = torch.tensor(a)
        result = torch.atan(input=torch.tensor(a), out=out)
        """
    )
    obj.run(pytorch_code, ["out", "result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = [0.2341, 0.2539, -0.6256, -0.6448]
        out = torch.tensor(a)
        result = torch.atan(out=out, input=torch.tensor(a))
        """
    )
    obj.run(pytorch_code, ["out", "result"])


def test_case_6():
    """2D张量"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[0.2341, 0.2539], [-0.6256, -0.6448]])
        result = torch.atan(a)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    """3D张量"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]])
        result = torch.atan(a)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    """边界值 - 零值和无穷大"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([0.0, 1.0, -1.0, 100.0, -100.0])
        result = torch.atan(a)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    """不同数据类型 - float64"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([0.2341, 0.2539, -0.6256], dtype=torch.float64)
        result = torch.atan(a)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_10():
    """梯度计算测试"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([0.5, 1.0, 2.0], requires_grad=True)
        y = torch.atan(a)
        y.sum().backward()
        a_grad = a.grad
        """
    )
    obj.run(pytorch_code, ["y", "a_grad"], check_stop_gradient=False)
