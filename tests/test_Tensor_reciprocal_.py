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

from apibase import APIBase
from inplace_unary_test_utils import run_torch_case

obj = APIBase("torch.Tensor.reciprocal_")


def test_case_1():
    run_torch_case(
        obj,
        """
        x = torch.tensor([0.5, -2.0, 4.0, -0.25], dtype=torch.float32)
        result = x.reciprocal_()
        """,
        ["x", "result"],
    )


def test_case_2():
    run_torch_case(
        obj,
        """
        x = torch.tensor([[0.125, -1.5], [3.0, -8.0]], dtype=torch.float64)
        result = x.reciprocal_()
        """,
        ["x", "result"],
    )


def test_case_3():
    run_torch_case(
        obj,
        """
        x = torch.tensor(
            [0.2, -0.4, 0.8, -1.6, 2.0, -2.5, 4.0, -5.0], dtype=torch.float32
        ).reshape(2, 2, 2)
        result = x.reciprocal_()
        """,
        ["x", "result"],
    )


def test_case_4():
    run_torch_case(
        obj,
        """
        result = torch.tensor([0.75, -1.25, 2.5], dtype=torch.float64).reciprocal_()
        """,
        ["result"],
    )
