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

obj = APIBase("torch.optim.Adadelta")


def test_case_1():
    pytorch_code = textwrap.dedent(
        generate_optimizer_test_code("torch.optim.Adadelta(conv.parameters())")
    )
    obj.run(pytorch_code, ["result"], rtol=1.0e-5)


def test_case_2():
    pytorch_code = textwrap.dedent(
        generate_optimizer_test_code(
            "torch.optim.Adadelta(conv.parameters(),weight_decay=0.01)"
        )
    )
    obj.run(pytorch_code, ["result"], rtol=1.0e-5)


def test_case_3():
    pytorch_code = textwrap.dedent(
        generate_optimizer_test_code(
            "torch.optim.Adadelta(conv.parameters(),weight_decay=0.01,lr=0.1)"
        )
    )
    obj.run(pytorch_code, ["result"], rtol=1.0e-5)


def test_case_4():
    pytorch_code = textwrap.dedent(
        generate_optimizer_test_code(
            "torch.optim.Adadelta(conv.parameters(), 1.0, 0.9, 1e-6, 0.)"
        )
    )
    obj.run(pytorch_code, ["result"], rtol=1.0e-5)


def test_case_5():
    pytorch_code = textwrap.dedent(
        generate_optimizer_test_code(
            "torch.optim.Adadelta(params=conv.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0.)"
        )
    )
    obj.run(pytorch_code, ["result"], rtol=1.0e-5)


def test_case_6():
    pytorch_code = textwrap.dedent(
        generate_optimizer_test_code(
            "torch.optim.Adadelta(conv.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0, foreach=None, maximize=False, differentiable=False)"
        )
    )
    obj.run(
        pytorch_code,
        unsupport=True,
        reason="param `foreach`, `maximize` and `differentiable` is not supported.",
    )


def test_case_7():
    pytorch_code = textwrap.dedent(
        generate_optimizer_test_code(
            "torch.optim.Adadelta(params=conv.parameters(), weight_decay=0)"
        )
    )
    obj.run(pytorch_code, ["result"], rtol=1.0e-5)
