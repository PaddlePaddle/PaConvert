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

obj = APIBase("torch.optim.lr_scheduler.ConstantLR")


def test_case_1():
    pytorch_code = textwrap.dedent(
        generate_torch_code("torch.optim.lr_scheduler.ConstantLR(sgd, verbose=True)")
    )
    obj.run(pytorch_code, ["result1", "result2"], rtol=1.0e-5)


def test_case_2():
    pytorch_code = textwrap.dedent(
        generate_torch_code("torch.optim.lr_scheduler.ConstantLR(sgd, factor=0.05)")
    )
    obj.run(pytorch_code, ["result1", "result2"], rtol=1.0e-5)


def test_case_3():
    pytorch_code = textwrap.dedent(
        generate_torch_code("torch.optim.lr_scheduler.ConstantLR(sgd, total_iters=3)")
    )
    obj.run(pytorch_code, ["result1", "result2"], rtol=1.0e-5)


def test_case_4():
    pytorch_code = textwrap.dedent(
        generate_torch_code("torch.optim.lr_scheduler.ConstantLR(sgd, 0.05, 3)")
    )
    obj.run(pytorch_code, ["result1", "result2"], rtol=1.0e-5)


def test_case_5():
    pytorch_code = textwrap.dedent(
        generate_torch_code(
            "torch.optim.lr_scheduler.ConstantLR(optimizer=sgd, factor=0.05, total_iters=3)"
        )
    )
    obj.run(pytorch_code, ["result1", "result2"], rtol=1.0e-5)


def test_case_6():
    pytorch_code = textwrap.dedent(
        generate_torch_code(
            "torch.optim.lr_scheduler.ConstantLR(factor=0.05, total_iters=3, optimizer=sgd)"
        )
    )
    obj.run(pytorch_code, ["result1", "result2"], rtol=1.0e-5)


def test_case_7():
    pytorch_code = textwrap.dedent(
        generate_torch_code(
            "torch.optim.lr_scheduler.ConstantLR(sgd, 0.05, 3, -1, False)"
        )
    )
    obj.run(pytorch_code, ["result1", "result2"], rtol=1.0e-5)


def test_case_8():
    pytorch_code = textwrap.dedent(
        generate_torch_code(
            "torch.optim.lr_scheduler.ConstantLR(optimizer=sgd, factor=0.05, total_iters=3, last_epoch=-1, verbose=False)"
        )
    )
    obj.run(pytorch_code, ["result1", "result2"], rtol=1.0e-5)


def test_case_9():
    pytorch_code = textwrap.dedent(
        generate_torch_code(
            [
                "torch.optim.lr_scheduler.ConstantLR(optimizer=sgd, factor=0.05, total_iters=3, last_epoch=-1, verbose=False)",
                "torch.optim.lr_scheduler.ConstantLR(optimizer=sgd, factor=0.05, total_iters=3, last_epoch=scheduler_1.last_epoch, verbose=False)",
            ]
        )
    )
    obj.run(pytorch_code, ["result1", "result2"], rtol=1.0e-5)


def test_case_10():
    pytorch_code = textwrap.dedent(
        generate_torch_code("torch.optim.lr_scheduler.ConstantLR(sgd)")
    )
    obj.run(pytorch_code, ["result1", "result2"], rtol=1.0e-5)
