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
import typing

from apibase import APIBase


class ScheduleAPIBase(APIBase):
    def compare(
        self,
        name,
        pytorch_result,
        paddle_result,
        check_value=True,
        check_dtype=True,
        check_stop_gradient=True,
        rtol=1.0e-6,
        atol=0.0,
    ):
        assert isinstance(paddle_result, typing.Callable)


obj = ScheduleAPIBase("torch.profiler.schedule")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.profiler.schedule(wait=1, warmup=1, active=1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.profiler.schedule(wait=1, active=2, warmup=3)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.profiler.schedule(wait=1, warmup=2, active=1, repeat=0, skip_first=0)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.profiler.schedule(wait=1, warmup=2, active=1, repeat=1, skip_first=0)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.profiler.schedule(wait=1, warmup=2, active=1, repeat=0, skip_first=1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.profiler.schedule(warmup=2, repeat=1, active=1, wait=1, skip_first=0)
        """
    )
    obj.run(pytorch_code, ["result"])