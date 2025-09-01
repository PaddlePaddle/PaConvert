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

obj = APIBase("torch.Tensor.new_zeros")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1., 2., 3.])
        result = x.new_zeros((1,))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1., 2., 3.])
        result = x.new_zeros((1, 3), dtype=torch.float64)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1., 2., 3.])
        result = x.new_zeros((3, 4), device='cpu')
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1., 2., 3.])
        result = x.new_zeros((5, 7), device='cpu', requires_grad=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1., 2., 3.])
        result = x.new_zeros((2, 3, 4), device='cpu', requires_grad=False)
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skip("skip util some attr transform fixed(e.g. layout)")
def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1., 2., 3.])
        result = x.new_zeros((7, 2, 1, 9), dtype=torch.float64, device='cpu', requires_grad=True, layout=torch.strided)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1., 2., 3.])
        result = x.new_zeros(size=(2, 3), dtype=torch.float64, device='cpu', requires_grad=True, pin_memory=False)
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skip("skip util some attr transform fixed(e.g. layout)")
def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1., 2., 3.])
        result = x.new_zeros(size=(2, 3), dtype=torch.float64, device='cpu', requires_grad=True, layout=torch.strided, pin_memory=False)
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skip("skip util some attr transform fixed(e.g. layout)")
def test_case_9():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1., 2., 3.])
        result = x.new_zeros(layout=torch.strided, dtype=torch.float64, size=(2, 3), device='cpu', requires_grad=True, pin_memory=False)
        """
    )
    obj.run(pytorch_code, ["result"])
