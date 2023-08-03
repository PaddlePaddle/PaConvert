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

obj = APIBase("torch.linalg.cond")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1., 0, -1], [0, 1, 0], [1, 0, 1]])
        result = torch.linalg.cond(x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1., 0, -1], [0, 1, 0], [1, 0, 1]])
        result = torch.linalg.cond(input=x, p='fro')
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1., 0, -1], [0, 1, 0], [1, 0, 1]])
        result = torch.linalg.cond(p=1, input=x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1., 0, -1], [0, 1, 0], [1, 0, 1]])
        out = torch.tensor([])
        result = torch.linalg.cond(x, float('inf'), out=out)
        """
    )
    obj.run(pytorch_code, ["result", "out"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1., 0, -1], [0, 1, 0], [1, 0, 1]])
        out = torch.tensor([])
        result = torch.linalg.cond(input=x, p=float('inf'), out=out)
        """
    )
    obj.run(pytorch_code, ["result", "out"])
