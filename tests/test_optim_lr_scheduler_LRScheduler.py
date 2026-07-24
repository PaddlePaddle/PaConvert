# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

obj = APIBase("torch.optim.lr_scheduler.LRScheduler")


def generate_test_code(scheduler_init, prelude=""):
    return f"""
        import torch

        class ConstantScheduler(torch.optim.lr_scheduler.LRScheduler):
            def get_lr(self):
                if hasattr(self, "base_lrs"):
                    return [base_lr * 0.5 for base_lr in self.base_lrs]
                return self.base_lr * 0.5

        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        {prelude}
        scheduler = {scheduler_init}
        loss = parameter.sum()
        loss.backward()
        optimizer.step()
        result = parameter
        result_epoch = scheduler.last_epoch
        result_is_base = isinstance(
            scheduler, torch.optim.lr_scheduler.LRScheduler
        )
    """


def run_test(scheduler_init, prelude=""):
    pytorch_code = textwrap.dedent(generate_test_code(scheduler_init, prelude))
    obj.run(pytorch_code, ["result", "result_epoch", "result_is_base"])


def test_case_1():
    """Optimizer positional argument with default last_epoch."""
    run_test("ConstantScheduler(optimizer)")


def test_case_2():
    """Optimizer keyword argument with default last_epoch."""
    run_test("ConstantScheduler(optimizer=optimizer)")


def test_case_3():
    """All arguments passed positionally."""
    run_test("ConstantScheduler(optimizer, -1)")


def test_case_4():
    """Mixed positional and keyword arguments."""
    run_test("ConstantScheduler(optimizer, last_epoch=-1)")


def test_case_5():
    """Keyword arguments passed out of order."""
    run_test("ConstantScheduler(last_epoch=-1, optimizer=optimizer)")


def test_case_6():
    """Variable positional arguments."""
    run_test("ConstantScheduler(*args)", "args = (optimizer,)")


def test_case_7():
    """Variable keyword arguments."""
    run_test(
        "ConstantScheduler(**kwargs)",
        'kwargs = {"last_epoch": -1, "optimizer": optimizer}',
    )
