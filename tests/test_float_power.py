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

obj = APIBase("torch.float_power")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([4, 6, 7, 1])
        result = torch.float_power(x, 2)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([4, 6, 7, 1])
        out = torch.zeros([4, 1], dtype=torch.double)
        result = torch.float_power(x, 2, out=out)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([4, 6, 7, 1])
        y = torch.tensor([2, -3, 4, -5])
        result = torch.float_power(x, y)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([4, 6, 7, 1])
        y = torch.tensor([2, -3, 4, -5])
        out = torch.zeros([4, 1], dtype=torch.double)
        result = torch.float_power(x, y, out=out)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, -2], [2, 5]])
        result = torch.float_power(x, 2)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1., -2.], [2., 5.]])
        out = torch.zeros([2, 2], dtype=torch.double)
        result = torch.float_power(x, 2, out=out)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, -2], [2, 5]])
        y = torch.tensor([[-2, 3], [-1, 2]])
        out = torch.zeros([2, 2], dtype=torch.double)
        result = torch.float_power(x, y, out=out)
        """
    )
    obj.run(pytorch_code, ["result"])
