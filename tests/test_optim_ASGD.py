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
from optimizer_helper import generate_optimizer_test_code

obj = APIBase("torch.optim.Adam")


def test_case_1():
    pytorch_code = textwrap.dedent(
        generate_optimizer_test_code(
            "torch.optim.ASGD(params=conv.parameters(), lr=0.01, weight_decay=0.0001)"
        )
    )
    obj.run(pytorch_code, ["result"], rtol=1.0e-5)


def test_case_2():
    pytorch_code = textwrap.dedent(
        generate_optimizer_test_code(
            "torch.optim.ASGD(params=conv.parameters(), lambd=1e-3, alpha=0.8, t0=1e5, lr=0.01, weight_decay=0.0001, foreach=False, differentiable=False)"
        )
    )
    obj.run(pytorch_code, ["result"], rtol=1.0e-5)


def test_case_3():
    pytorch_code = textwrap.dedent(
        generate_optimizer_test_code("torch.optim.ASGD(conv.parameters())")
    )
    obj.run(pytorch_code, ["result"], rtol=1.0e-1)


def test_case_4():
    pytorch_code = textwrap.dedent(
        generate_optimizer_test_code(
            "torch.optim.ASGD(params=conv.parameters(), lambd=1e-3, alpha=0.8, t0=1e5, lr=0.01, weight_decay=0.0001, maximize=True, differentiable=False)"
        )
    )
    obj.run(
        pytorch_code,
        ["result"],
        rtol=1.0e-5,
        unsupport=True,
        reason="maximize is not supported yet.",
    )


def test_case_5():
    pytorch_code = textwrap.dedent(
        generate_optimizer_test_code(
            "torch.optim.ASGD(weight_decay=0.0001, lambd=1e-3, alpha=0.8, t0=1e5, lr=0.01, foreach=False, params=conv.parameters(), differentiable=False)"
        )
    )
    obj.run(pytorch_code, ["result"], rtol=1.0e-5)
