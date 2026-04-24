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
from unary_inplace_test_utils import register_standard_unary_inplace_tests

obj = APIBase("torch.Tensor.exp_")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.tensor([0., -2., 3.]).exp_()
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([-1., -2., 3.])
        result = a.exp_()
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[0.25, -1.5], [2.2, -0.75]], dtype=torch.float64)
        result = a.exp_()
        """
    )
    obj.run(pytorch_code, ["result", "a"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor(
            [
                [[0.2, -0.7], [1.5, -1.1]],
                [[0.9, -0.4], [2.3, -0.2]],
            ],
            dtype=torch.float32,
        )
        result = a.exp_()
        """
    )
    obj.run(pytorch_code, ["result", "a"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[-0.6, 0.4, 1.2], [0.7, -1.3, 2.1]], dtype=torch.float32)
        alias = a
        result = alias.exp_()
        """
    )
    obj.run(pytorch_code, ["result", "a", "alias"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        base = torch.tensor([[0.3, -1.4, 0.9], [1.7, -0.8, 0.5]], dtype=torch.float32)
        a = base.t()
        result = a.exp_()
        """
    )
    obj.run(pytorch_code, ["result", "a", "base"])


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.empty([0], dtype=torch.float32)
        result = a.exp_()
        """
    )
    obj.run(pytorch_code, ["result", "a"])


register_standard_unary_inplace_tests(
    globals(),
    obj,
    "exp_",
    "[[-0.6, 0.4, 1.2], [0.7, -1.3, 2.1]]",
    "[[[0.2, -0.7], [1.5, -1.1]], [[0.9, -0.4], [2.3, -0.2]]]",
)
