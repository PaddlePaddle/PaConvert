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

obj = APIBase("torch.optim.Adamax")


def test_case_1():
    pytorch_code = textwrap.dedent(
        generate_optimizer_test_code(
            "torch.optim.Adamax(conv.parameters(), betas=(0.5, 0.99))"
        )
    )
    obj.run(pytorch_code, ["result"], rtol=1.0e-2)


def test_case_2():
    pytorch_code = textwrap.dedent(
        generate_optimizer_test_code("torch.optim.Adamax(conv.parameters(), lr=0.06)")
    )
    obj.run(pytorch_code, ["result"], rtol=1.0e-2)


def test_case_3():
    pytorch_code = textwrap.dedent(
        generate_optimizer_test_code(
            "torch.optim.Adamax(conv.parameters(), weight_decay=0.1, eps=1e-7)"
        )
    )
    obj.run(pytorch_code, ["result"], rtol=1.0e-2)


def test_case_4():
    pytorch_code = textwrap.dedent(
        generate_optimizer_test_code(
            "torch.optim.Adamax(params=conv.parameters(), lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.)"
        )
    )
    obj.run(pytorch_code, ["result"], rtol=1.0e-2)


def test_case_5():
    pytorch_code = textwrap.dedent(
        generate_optimizer_test_code(
            "torch.optim.Adamax(conv.parameters(), 0.002, (0.9, 0.999), 1e-08, 0.)"
        )
    )
    obj.run(pytorch_code, ["result"], rtol=1.0e-2)


def test_case_6():
    pytorch_code = textwrap.dedent(
        generate_optimizer_test_code("torch.optim.Adamax(conv.parameters())")
    )
    obj.run(pytorch_code, ["result"], rtol=1.0e-2)
