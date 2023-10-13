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
from lr_scheduler_helper import generate_torch_code

obj = APIBase("torch.optim.lr_scheduler.MultiplicativeLR")


def test_case_1():
    pytorch_code = textwrap.dedent(
        generate_torch_code(
            "torch.optim.lr_scheduler.MultiplicativeLR(sgd, lambda x:0.95**x)"
        )
    )
    obj.run(pytorch_code, ["result1", "result2"], rtol=1.0e-5)


def test_case_2():
    pytorch_code = textwrap.dedent(
        generate_torch_code(
            "torch.optim.lr_scheduler.MultiplicativeLR(sgd, lr_lambda=lambda x:0.95**x)"
        )
    )
    obj.run(pytorch_code, ["result1", "result2"], rtol=1.0e-5)


def test_case_3():
    pytorch_code = textwrap.dedent(
        generate_torch_code(
            "torch.optim.lr_scheduler.MultiplicativeLR(optimizer=sgd, lr_lambda=lambda x:0.95**x)"
        )
    )
    obj.run(pytorch_code, ["result1", "result2"], rtol=1.0e-5)


def test_case_4():
    pytorch_code = textwrap.dedent(
        generate_torch_code(
            "torch.optim.lr_scheduler.MultiplicativeLR(optimizer=sgd, lr_lambda=lambda x:0.95**x, last_epoch=-1, verbose=True)"
        )
    )
    obj.run(pytorch_code, ["result1", "result2"], rtol=1.0e-5)


def test_case_5():
    pytorch_code = textwrap.dedent(
        generate_torch_code(
            "torch.optim.lr_scheduler.MultiplicativeLR(optimizer=sgd, lr_lambda=lambda x:0.95**x, verbose=True)"
        )
    )
    obj.run(pytorch_code, ["result1", "result2"], rtol=1.0e-5)


def test_case_6():
    pytorch_code = textwrap.dedent(
        generate_torch_code(
            "torch.optim.lr_scheduler.MultiplicativeLR(sgd, lambda x:0.95**x, -1, False)"
        )
    )
    obj.run(pytorch_code, ["result1", "result2"], rtol=1.0e-5)
