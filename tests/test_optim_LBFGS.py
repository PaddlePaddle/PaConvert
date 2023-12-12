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

obj = APIBase("torch.optim.LBFGS")


def test_case_1():
    pytorch_code = textwrap.dedent(
        generate_optimizer_test_code(
            "torch.optim.LBFGS(conv.parameters(), max_iter=20)",
            step_with_closure=True,
        )
    )
    obj.run(pytorch_code, ["result"], rtol=1.0e-3)


def test_case_2():
    pytorch_code = textwrap.dedent(
        generate_optimizer_test_code(
            "torch.optim.LBFGS(conv.parameters(), max_iter=40, lr=0.5)",
            step_with_closure=True,
        )
    )
    obj.run(pytorch_code, ["result"], rtol=1.0e-2)


def test_case_3():
    pytorch_code = textwrap.dedent(
        generate_optimizer_test_code(
            "torch.optim.LBFGS(conv.parameters(), max_iter=30)",
            step_with_closure=True,
        )
    )
    obj.run(pytorch_code, ["result"], rtol=1.0e-3)


def test_case_4():
    pytorch_code = textwrap.dedent(
        generate_optimizer_test_code(
            "torch.optim.LBFGS(conv.parameters(), max_iter=30)",
            step_with_closure=True,
        )
    )
    obj.run(pytorch_code, ["result"], rtol=1.0e-3)


def test_case_5():
    pytorch_code = textwrap.dedent(
        generate_optimizer_test_code(
            "torch.optim.LBFGS(conv.parameters())",
            step_with_closure=True,
        )
    )
    obj.run(pytorch_code, ["result"], rtol=1.0e-3)


def test_case_6():
    pytorch_code = textwrap.dedent(
        generate_optimizer_test_code(
            "torch.optim.LBFGS(params=conv.parameters(), lr=1, max_iter=20, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=100, line_search_fn=None)",
            step_with_closure=True,
        )
    )
    obj.run(pytorch_code, ["result"], rtol=1.0e-3)


def test_case_7():
    pytorch_code = textwrap.dedent(
        generate_optimizer_test_code(
            "torch.optim.LBFGS(conv.parameters(), 1, 20, None, 1e-07, 1e-09, 100, None)",
            step_with_closure=True,
        )
    )
    obj.run(pytorch_code, ["result"], rtol=1.0e-3)


def test_case_8():
    pytorch_code = textwrap.dedent(
        generate_optimizer_test_code(
            "torch.optim.LBFGS(lr=1, max_iter=20, params=conv.parameters(), tolerance_grad=1e-07, tolerance_change=1e-09, history_size=100, line_search_fn=None, max_eval=None)",
            step_with_closure=True,
        )
    )
    obj.run(pytorch_code, ["result"], rtol=1.0e-3)
