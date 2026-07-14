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

import pytest
from apibase import APIBase

obj = APIBase("torch.orgqr")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[[-1.0, 2.0], [3.0, -4.0]], [[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float64)
        tau = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
        result = torch.orgqr(a, tau)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[[-1.0, 2.0], [3.0, -4.0]]], dtype=torch.float64)
        tau = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
        result = torch.orgqr(a, tau)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[-1.0, 2.0], [3.0, -4.0]], dtype=torch.float64)
        tau = torch.tensor([1.0, 0.0], dtype=torch.float64)
        result = torch.orgqr(a, tau)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
        tau = torch.tensor([0.0, 0.0], dtype=torch.float64)
        result = torch.orgqr(a, tau)
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skip(
    reason="Paddle orgqr does not support variable argument unpacking (*args), PyTorch supports it"
)
def test_case_5():
    """Variable argument test"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[-1.0, 2.0], [3.0, -4.0]], dtype=torch.float64)
        tau = torch.tensor([1.0, 0.0], dtype=torch.float64)
        args = (a, tau)
        result = torch.orgqr(*args)
        """
    )
    obj.run(pytorch_code, ["result"])
