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

obj = APIBase("torch.ceil_")


def test_case_1():
    run_torch_case(
        obj,
        """
        x = torch.tensor([1.2, -3.4, 0.1, 2.0], dtype=torch.float32)
        result = torch.ceil_(x)
        """,
        ["x", "result"],
    )


def test_case_2():
    run_torch_case(
        obj,
        """
        x = torch.tensor([[-1.2, 1.01], [2.5, -4.75]], dtype=torch.float64)
        result = torch.ceil_(input=x)
        """,
        ["x", "result"],
    )


def test_case_3():
    run_torch_case(
        obj,
        """
        x = torch.tensor(
            [-2.3, -1.1, -0.01, 0.2, 1.3, 2.8, 3.01, 4.5], dtype=torch.float32
        ).reshape(2, 2, 2)
        args = (x,)
        result = torch.ceil_(*args)
        """,
        ["x", "result"],
    )


def test_case_4():
    run_torch_case(
        obj,
        """
        result = torch.ceil_(torch.tensor([-2.2, 0.0, 3.3], dtype=torch.float64))
        """,
        ["result"],
    )
