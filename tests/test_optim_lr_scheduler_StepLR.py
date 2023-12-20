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
from lr_scheduler_helper import generate_lr_scheduler_test_code

obj = APIBase("torch.optim.lr_scheduler.StepLR")


def test_case_1():
    pytorch_code = textwrap.dedent(
        generate_lr_scheduler_test_code(
            "torch.optim.lr_scheduler.StepLR(sgd, step_size=2, verbose=True)"
        )
    )
    obj.run(pytorch_code, ["result1", "result2"], rtol=1.0e-5)


def test_case_2():
    pytorch_code = textwrap.dedent(
        generate_lr_scheduler_test_code(
            "torch.optim.lr_scheduler.StepLR(sgd, step_size=2, gamma=0.05)"
        )
    )
    obj.run(pytorch_code, ["result1", "result2"], rtol=1.0e-5)


def test_case_3():
    pytorch_code = textwrap.dedent(
        generate_lr_scheduler_test_code(
            "torch.optim.lr_scheduler.StepLR(sgd, step_size=3, gamma=0.2)"
        )
    )
    obj.run(pytorch_code, ["result1", "result2"], rtol=1.0e-5)


def test_case_4():
    pytorch_code = textwrap.dedent(
        generate_lr_scheduler_test_code("torch.optim.lr_scheduler.StepLR(sgd, 10, 0.2)")
    )
    obj.run(pytorch_code, ["result1", "result2"], rtol=1.0e-5)


def test_case_5():
    pytorch_code = textwrap.dedent(
        generate_lr_scheduler_test_code(
            "torch.optim.lr_scheduler.StepLR(sgd, 10, 0.2, -1, False)"
        )
    )
    obj.run(pytorch_code, ["result1", "result2"], rtol=1.0e-5)


def test_case_6():
    pytorch_code = textwrap.dedent(
        generate_lr_scheduler_test_code(
            "torch.optim.lr_scheduler.StepLR(optimizer=sgd, step_size=3, gamma=0.2)"
        )
    )
    obj.run(pytorch_code, ["result1", "result2"], rtol=1.0e-5)


def test_case_7():
    pytorch_code = textwrap.dedent(
        generate_lr_scheduler_test_code(
            "torch.optim.lr_scheduler.StepLR(optimizer=sgd, step_size=3, gamma=0.2, last_epoch=-1, verbose=False)"
        )
    )
    obj.run(pytorch_code, ["result1", "result2"], rtol=1.0e-5)


def test_case_8():
    pytorch_code = textwrap.dedent(
        generate_lr_scheduler_test_code(
            "torch.optim.lr_scheduler.StepLR(step_size=3, gamma=0.2, optimizer=sgd)"
        )
    )
    obj.run(pytorch_code, ["result1", "result2"], rtol=1.0e-5)


# note: StepLR does not support resume training
# paddle result has diff with pytorch result
def test_case_9():
    pytorch_code = textwrap.dedent(
        generate_lr_scheduler_test_code(
            [
                "torch.optim.lr_scheduler.StepLR(optimizer=sgd, step_size=3, gamma=0.2, last_epoch=-1, verbose=False)",
                "torch.optim.lr_scheduler.StepLR(optimizer=sgd, step_size=3, gamma=0.2, last_epoch=scheduler_1.last_epoch, verbose=False)",
            ]
        )
    )
    obj.run(pytorch_code, ["result1", "result2"], rtol=1.0e-5, check_value=False)


def test_case_10():
    pytorch_code = textwrap.dedent(
        generate_lr_scheduler_test_code("torch.optim.lr_scheduler.StepLR(sgd, 2)")
    )
    obj.run(pytorch_code, ["result1", "result2"], rtol=1.0e-5)
