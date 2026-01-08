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

obj = APIBase("torch.nextafter")


def test_case_1():
    """基础用法 - 1D张量"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([1.0, 2.0, 3.0])
        other = torch.tensor([1.1, 2.1, 3.1])
        result = torch.nextafter(input, other)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    """位置参数"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([0.0, 1.0, 2.0])
        b = torch.tensor([0.5, 1.5, 2.5])
        result = torch.nextafter(a, b)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    """关键字参数"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([0.0, 1.0, 2.0])
        b = torch.tensor([0.5, 1.5, 2.5])
        result = torch.nextafter(input=a, other=b)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    """关键字参数乱序"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([0.0, 1.0, 2.0])
        b = torch.tensor([0.5, 1.5, 2.5])
        result = torch.nextafter(other=b, input=a)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    """out参数"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([1.0, 2.0, 3.0])
        other = torch.tensor([1.1, 2.1, 3.1])
        out = torch.empty_like(input)
        result = torch.nextafter(input, other, out=out)
        """
    )
    obj.run(pytorch_code, ["out", "result"])


def test_case_6():
    """out参数 - 关键字形式"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([1.0, 2.0, 3.0])
        other = torch.tensor([1.1, 2.1, 3.1])
        out = torch.empty_like(input)
        result = torch.nextafter(input, other, out=out)
        """
    )
    obj.run(pytorch_code, ["out", "result"])


def test_case_7():
    """2D张量"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        other = torch.tensor([[1.1, 2.1], [3.1, 4.1]])
        result = torch.nextafter(input, other)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    """3D张量"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        other = torch.tensor([[[1.1, 2.1], [3.1, 4.1]], [[5.1, 6.1], [7.1, 8.1]]])
        result = torch.nextafter(input, other)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    """边界值 - 零值"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([0.0, -0.0, 1.0])
        other = torch.tensor([0.1, -0.1, 1.1])
        result = torch.nextafter(input, other)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_10():
    """不同数据类型 - float64"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([1.0, 2.0], dtype=torch.float64)
        other = torch.tensor([1.1, 2.1], dtype=torch.float64)
        result = torch.nextafter(input, other)
        """
    )
    obj.run(pytorch_code, ["result"])
