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
from inplace_unary_helper import run_torch_case

obj = APIBase("torch.sigmoid_")


def test_case_1():
    run_torch_case(
        obj,
        """
        x = torch.tensor([-3.0, -0.5, 0.5, 2.0], dtype=torch.float32)
        result = torch.sigmoid_(x)
        """,
        ["x", "result"],
    )


def test_case_2():
    run_torch_case(
        obj,
        """
        x = torch.tensor([[-4.0, -1.25], [0.75, 3.0]], dtype=torch.float64)
        result = torch.sigmoid_(input=x)
        """,
        ["x", "result"],
    )


def test_case_3():
    run_torch_case(
        obj,
        """
        x = torch.tensor(
            [-5.0, -2.5, -1.0, -0.25, 0.25, 1.0, 2.5, 5.0], dtype=torch.float32
        ).reshape(2, 2, 2)
        args = (x,)
        result = torch.sigmoid_(*args)
        """,
        ["x", "result"],
    )


def test_case_4():
    run_torch_case(
        obj,
        """
        result = torch.sigmoid_(torch.tensor([-1.5, 0.0, 1.5], dtype=torch.float64))
        """,
        ["result"],
    )
