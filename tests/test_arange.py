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

obj = APIBase("torch.arange")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.arange(5)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.arange(5.)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.arange(1, 4)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.arange(1, 4, step=2)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.arange(1, 2.5, 0.5)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.arange(1, 2.5, 0.5, dtype=torch.float64, requires_grad=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        out = torch.rand([3], dtype=torch.float64)
        result = torch.arange(1, 2.5, 0.5, out=out, dtype=torch.float64)
        """
    )
    obj.run(pytorch_code, ["result", "out"])


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        out = torch.rand([3], dtype=torch.float64)
        result = torch.arange(start=1, end=2.5, step=0.5, out=out, dtype=torch.float64, layout=torch.strided, device=torch.device('cpu'), requires_grad=True)
        """
    )
    obj.run(pytorch_code, ["result", "out"])


def test_case_9():
    pytorch_code = textwrap.dedent(
        """
        import torch
        out = torch.rand([3], dtype=torch.float64)
        result = torch.arange(device=torch.device('cpu'), end=2.5, step=0.5, out=out, dtype=torch.float64, layout=torch.strided, start=1, requires_grad=True)
        """
    )
    obj.run(pytorch_code, ["result", "out"])
