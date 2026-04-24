# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

obj = APIBase("torch.tan_")


def test_case_1():
    run_torch_case(
        obj,
        """
        x = torch.tensor([-1.0, -0.5, 0.25, 1.0], dtype=torch.float32)
        result = torch.tan_(x)
        """,
        ["x", "result"],
    )


def test_case_2():
    run_torch_case(
        obj,
        """
        x = torch.tensor([[-1.2, -0.25], [0.25, 1.2]], dtype=torch.float64)
        result = torch.tan_(input=x)
        """,
        ["x", "result"],
    )


def test_case_3():
    run_torch_case(
        obj,
        """
        x = torch.tensor(
            [-1.3, -1.0, -0.6, -0.2, 0.2, 0.6, 1.0, 1.3], dtype=torch.float32
        ).reshape(2, 2, 2)
        args = (x,)
        result = torch.tan_(*args)
        """,
        ["x", "result"],
    )


def test_case_4():
    run_torch_case(
        obj,
        """
        result = torch.tan_(torch.tensor([-0.75, 0.0, 0.75], dtype=torch.float64))
        """,
        ["result"],
    )
