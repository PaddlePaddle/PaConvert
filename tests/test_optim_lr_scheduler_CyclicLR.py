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

obj = APIBase("torch.optim.lr_scheduler.CyclicLR")


def test_case_1():
    pytorch_code = textwrap.dedent(
        generate_torch_code("torch.optim.lr_scheduler.CyclicLR(sgd, 0.01, 0.1)")
    )
    obj.run(pytorch_code, ["result1", "result2"], rtol=1.0e-5, check_value=False)


def test_case_2():
    pytorch_code = textwrap.dedent(
        generate_torch_code(
            "torch.optim.lr_scheduler.CyclicLR(sgd, base_lr=0.01, max_lr=0.1)"
        )
    )
    obj.run(pytorch_code, ["result1", "result2"], rtol=1.0e-5, check_value=False)


def test_case_3():
    pytorch_code = textwrap.dedent(
        generate_torch_code(
            "torch.optim.lr_scheduler.CyclicLR(optimizer=sgd, base_lr=0.01, max_lr=0.1)"
        )
    )
    obj.run(pytorch_code, ["result1", "result2"], rtol=1.0e-5, check_value=False)


def test_case_4():
    pytorch_code = textwrap.dedent(
        generate_torch_code(
            "torch.optim.lr_scheduler.CyclicLR(optimizer=sgd, base_lr=0.01, max_lr=0.1, step_size_up=1000, step_size_down=1000, mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle', cycle_momentum=True, base_momentum=0.8, max_momentum=0.9, last_epoch=-1, verbose=False)"
        )
    )
    obj.run(
        pytorch_code,
        ["result1", "result2"],
        rtol=1.0e-5,
        unsupport=True,
        reason="`cycle_momentum`, `base_momentum`, `max_momentum` is not supported.",
    )


def test_case_5():
    pytorch_code = textwrap.dedent(
        generate_torch_code(
            "torch.optim.lr_scheduler.CyclicLR(optimizer=sgd, base_lr=0.01, max_lr=0.1, step_size_up=1000, step_size_down=1000, mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle', last_epoch=-1, verbose=False)"
        )
    )
    obj.run(pytorch_code, ["result1", "result2"], rtol=1.0e-5, check_value=False)


def test_case_6():
    pytorch_code = textwrap.dedent(
        generate_torch_code(
            "torch.optim.lr_scheduler.CyclicLR(scale_fn=None, scale_mode='cycle', last_epoch=-1, verbose=False, base_lr=0.01, max_lr=0.1, step_size_up=1000, optimizer=sgd, mode='triangular', gamma=1.0)"
        )
    )
    obj.run(pytorch_code, ["result1", "result2"], rtol=1.0e-5, check_value=False)


def test_case_7():
    pytorch_code = textwrap.dedent(
        generate_torch_code(
            "torch.optim.lr_scheduler.CyclicLR(sgd, 0.01, 0.1, 1000, 1000, 'triangular', 1.0, None, 'cycle', last_epoch=-1, verbose=False)"
        )
    )
    obj.run(pytorch_code, ["result1", "result2"], rtol=1.0e-5, check_value=False)


def test_case_8():
    pytorch_code = textwrap.dedent(
        generate_torch_code(
            "torch.optim.lr_scheduler.CyclicLR(sgd, 0.01, 0.1, 1000, 1000, 'triangular', 1.0, None, 'cycle')"
        )
    )
    obj.run(pytorch_code, ["result1", "result2"], rtol=1.0e-5, check_value=False)
