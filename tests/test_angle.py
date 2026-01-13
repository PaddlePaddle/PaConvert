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

obj = APIBase("torch.angle")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.angle(torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j]))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j])
        result = torch.angle(x) * 180 / 3.14159
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j])
        out = torch.tensor([2., 3., 4.])
        result = torch.angle(x, out=out)
        """
    )
    obj.run(pytorch_code, ["result", "out"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j])
        out = torch.tensor([2., 3., 4.])
        result = torch.angle(input=x, out=out)
        """
    )
    obj.run(pytorch_code, ["result", "out"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j])
        out = torch.tensor([2., 3., 4.])
        result = torch.angle(out=out, input=x)
        """
    )
    obj.run(pytorch_code, ["result", "out"])


def test_case_6():
    """2D复数张量"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1+1j, 2+2j], [3+3j, 4+4j]])
        result = torch.angle(x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    """3D复数张量"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[[1+1j, 2+2j], [3+3j, 4+4j]], [[5+5j, 6+6j], [7+7j, 8+8j]]])
        result = torch.angle(x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    """实数输入"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1.0, 2.0, 3.0])
        result = torch.angle(x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    """零值复数"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([0+0j, 1+0j, 0+1j])
        result = torch.angle(x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_10():
    """不同数据类型 - float64"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1+1j, 2+2j, 3+3j], dtype=torch.complex128)
        result = torch.angle(x)
        """
    )
    obj.run(pytorch_code, ["result"])
