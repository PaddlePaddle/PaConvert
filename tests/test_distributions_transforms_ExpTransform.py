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

obj = APIBase("torch.distributions.transforms.ExpTransform")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch

        exp = torch.distributions.transforms.ExpTransform()
        result = exp.forward_shape([1, 2])
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch

        exp = torch.distributions.transforms.ExpTransform(cache_size=1)
        result = exp.forward_shape([1, 2])
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch

        exp = torch.distributions.transforms.ExpTransform(1)
        result = exp.forward_shape([1, 2])
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch

        exp = torch.distributions.transforms.ExpTransform(cache_size=0)
        result = exp.forward_shape([2, 3])
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    """Mixed arguments test"""
    pytorch_code = textwrap.dedent(
        """
        import torch

        exp = torch.distributions.transforms.ExpTransform(1)
        x = torch.tensor([1.0, 2.0, 3.0])
        result = exp(x)
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skip(
    reason="Paddle ExpTransform does not support .inv() method, PyTorch supports it"
)
def test_case_6():
    """Inverse operation test"""
    pytorch_code = textwrap.dedent(
        """
        import torch

        exp = torch.distributions.transforms.ExpTransform(cache_size=1)
        x = torch.tensor([1.0, 2.0, 3.0])
        result = exp.inv(x)
        """
    )
    obj.run(pytorch_code, ["result"])
