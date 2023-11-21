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

obj = APIBase("torch.optim.lr_scheduler.OneCycleLR")


def test_case_1():
    pytorch_code = textwrap.dedent(
        generate_torch_code("torch.optim.lr_scheduler.OneCycleLR(sgd, 0.01, 100)")
    )
    obj.run(pytorch_code, ["result1", "result2"], rtol=1.0e-5, check_value=False)


def test_case_2():
    pytorch_code = textwrap.dedent(
        generate_torch_code(
            "torch.optim.lr_scheduler.OneCycleLR(sgd, max_lr=0.01, total_steps=200)"
        )
    )
    obj.run(pytorch_code, ["result1", "result2"], rtol=1.0e-5, check_value=False)


def test_case_3():
    pytorch_code = textwrap.dedent(
        generate_torch_code(
            "torch.optim.lr_scheduler.OneCycleLR(optimizer=sgd, max_lr=0.01, steps_per_epoch=20, epochs=10)"
        )
    )
    obj.run(pytorch_code, ["result1", "result2"], rtol=1.0e-5, check_value=False)


def test_case_4():
    pytorch_code = textwrap.dedent(
        generate_torch_code(
            "torch.optim.lr_scheduler.OneCycleLR(optimizer=sgd, max_lr=0.01, steps_per_epoch=20, epochs=10, last_epoch=-1, verbose=False)"
        )
    )
    obj.run(pytorch_code, ["result1", "result2"], rtol=1.0e-5, check_value=False)


def test_case_5():
    pytorch_code = textwrap.dedent(
        generate_torch_code(
            "torch.optim.lr_scheduler.OneCycleLR(optimizer=sgd, max_lr=0.01, steps_per_epoch=20, epochs=10, verbose=False)"
        )
    )
    obj.run(pytorch_code, ["result1", "result2"], rtol=1.0e-5, check_value=False)


def test_case_6():
    pytorch_code = textwrap.dedent(
        generate_torch_code(
            "torch.optim.lr_scheduler.OneCycleLR(optimizer=sgd, max_lr=0.01, total_steps=None, steps_per_epoch=20, epochs=10, pct_start=0.3, anneal_strategy='cos', div_factor=25.0, final_div_factor=10000.0, three_phase=False, last_epoch=-1, verbose=False)"
        )
    )
    obj.run(pytorch_code, ["result1", "result2"], rtol=1.0e-5, check_value=False)


def test_case_7():
    pytorch_code = textwrap.dedent(
        generate_torch_code(
            "torch.optim.lr_scheduler.OneCycleLR(optimizer=sgd, max_lr=0.01, total_steps=None, steps_per_epoch=20, epochs=10, pct_start=0.3, anneal_strategy='cos', cycle_momentum=True, base_momentum=0.85, max_momentum=0.95, div_factor=25.0, final_div_factor=10000.0, three_phase=False, last_epoch=-1, verbose=False)"
        )
    )
    obj.run(
        pytorch_code,
        ["result1", "result2"],
        rtol=1.0e-5,
        unsupport=True,
        reason="`cycle_momentum`, `base_momentum`, `max_momentum` is not supported.",
    )


def test_case_8():
    pytorch_code = textwrap.dedent(
        generate_torch_code(
            "torch.optim.lr_scheduler.OneCycleLR(optimizer=sgd, max_lr=0.01, steps_per_epoch=20, epochs=10, pct_start=0.3, anneal_strategy='linear', div_factor=2.5, final_div_factor=2000.0, three_phase=False, last_epoch=-1, verbose=False)"
        )
    )
    obj.run(pytorch_code, ["result1", "result2"], rtol=1.0e-5, check_value=False)


def test_case_9():
    pytorch_code = textwrap.dedent(
        generate_torch_code(
            "torch.optim.lr_scheduler.OneCycleLR(sgd, 0.01, None, 20, 10, 0.3, 'linear', div_factor=2.5, final_div_factor=2000.0, three_phase=False, last_epoch=-1, verbose=False)"
        )
    )
    obj.run(pytorch_code, ["result1", "result2"], rtol=1.0e-5, check_value=False)


def test_case_10():
    pytorch_code = textwrap.dedent(
        generate_torch_code(
            "torch.optim.lr_scheduler.OneCycleLR(sgd, 0.01, None, 20, 10, 0.3, 'linear')"
        )
    )
    obj.run(pytorch_code, ["result1", "result2"], rtol=1.0e-5, check_value=False)


def test_case_11():
    pytorch_code = textwrap.dedent(
        generate_torch_code(
            [
                "torch.optim.lr_scheduler.OneCycleLR(sgd, 0.01, None, 20, 10, 0.3, 'linear', div_factor=2.5, final_div_factor=2000.0, three_phase=False, last_epoch=-1, verbose=False)",
                "torch.optim.lr_scheduler.OneCycleLR(sgd, 0.01, None, 20, 10, 0.3, 'linear', div_factor=2.5, final_div_factor=2000.0, three_phase=False, last_epoch=scheduler_1.last_epoch, verbose=False)",
            ]
        )
    )
    obj.run(pytorch_code, ["result1", "result2"], rtol=1.0e-5, check_value=False)
