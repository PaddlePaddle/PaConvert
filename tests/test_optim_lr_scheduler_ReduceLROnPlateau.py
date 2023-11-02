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

obj = APIBase("torch.optim.lr_scheduler.ReduceLROnPlateau")


def test_case_1():
    pytorch_code = textwrap.dedent(
        generate_torch_code(
            "torch.optim.lr_scheduler.ReduceLROnPlateau(sgd)", step_with_loss=True
        )
    )
    obj.run(pytorch_code, ["result1", "result2"], rtol=1.0e-5)


def test_case_2():
    pytorch_code = textwrap.dedent(
        generate_torch_code(
            "torch.optim.lr_scheduler.ReduceLROnPlateau(sgd, 'min')",
            step_with_loss=True,
        )
    )
    obj.run(pytorch_code, ["result1", "result2"], rtol=1.0e-5)


def test_case_3():
    pytorch_code = textwrap.dedent(
        generate_torch_code(
            "torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=sgd, mode='min')",
            step_with_loss=True,
        )
    )
    obj.run(pytorch_code, ["result1", "result2"], rtol=1.0e-5)


def test_case_4():
    pytorch_code = textwrap.dedent(
        generate_torch_code(
            "torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=sgd, mode='min',verbose=True)",
            step_with_loss=True,
        )
    )
    obj.run(pytorch_code, ["result1", "result2"], rtol=1.0e-5)


def test_case_5():
    pytorch_code = textwrap.dedent(
        generate_torch_code(
            "torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=sgd, mode='min', factor=0.1, patience=10, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8, verbose=False)",
            step_with_loss=True,
        )
    )
    obj.run(pytorch_code, ["result1", "result2"], rtol=1.0e-5)


def test_case_6():
    pytorch_code = textwrap.dedent(
        generate_torch_code(
            "torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=sgd, mode='max', factor=0.1, patience=10, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8, verbose=False)",
            step_with_loss=True,
        )
    )
    obj.run(pytorch_code, ["result1", "result2"], rtol=1.0e-5)


def test_case_7():
    pytorch_code = textwrap.dedent(
        generate_torch_code(
            "torch.optim.lr_scheduler.ReduceLROnPlateau(sgd, 'max', 0.1, 10, 1e-4, 'rel', 0, 0, 1e-8, False)",
            step_with_loss=True,
        )
    )
    obj.run(pytorch_code, ["result1", "result2"], rtol=1.0e-5)


# note: fine, ReduceLROnPlateau does not support `last_epoch`
def test_case_8():
    pytorch_code = textwrap.dedent(
        generate_torch_code(
            [
                "torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=sgd, mode='min',verbose=False)",
                "torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=sgd, mode='min',verbose=True)",
            ],
            step_with_loss=True,
        )
    )
    obj.run(pytorch_code, ["result1", "result2"], rtol=1.0e-5)
