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

obj = APIBase("torch.linalg.pinv")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.arange(15).reshape((3, 5)).to(dtype=torch.float64)
        result = torch.linalg.pinv(x)
        """
    )
    obj.run(pytorch_code, ["result"], atol=1e-7)


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.arange(15).reshape((3, 5)).to(dtype=torch.float64)
        result = torch.linalg.pinv(input=x)
        """
    )
    obj.run(pytorch_code, ["result"], atol=1e-7)


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 2], [2, 1]]).to(dtype=torch.float32)
        out = torch.tensor([])
        result = torch.linalg.pinv(hermitian=True, input=x, out=out)
        """
    )
    obj.run(pytorch_code, ["result", "out"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 2], [2, 1]]).to(dtype=torch.float32)
        out = torch.tensor([])
        result = torch.linalg.pinv(input=x, atol=None, rtol=1e-5, hermitian=False, out=out)
        """
    )
    obj.run(pytorch_code, ["result", "out"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 2], [2, 1]]).to(dtype=torch.float32)
        out = torch.tensor([])
        result = torch.linalg.pinv(x, atol=None, rtol=1e-5, hermitian=False, out=out)
        """
    )
    obj.run(pytorch_code, ["result", "out"])
