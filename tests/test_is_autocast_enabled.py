# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
#

import textwrap

import paddle
import pytest
from apibase import APIBase

should_skip = not paddle.device.is_compiled_with_cuda()
skip_reason = "AMP test can only run with CUDA."

obj = APIBase("torch.is_autocast_enabled")


@pytest.mark.skipif(condition=should_skip, reason=skip_reason)
def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.is_autocast_enabled()
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skipif(condition=should_skip, reason=skip_reason)
def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result_before = torch.is_autocast_enabled()
        with torch.autocast(device_type='cuda', enabled=True):
            result_inside = torch.is_autocast_enabled()
        result_after = torch.is_autocast_enabled()
        """
    )
    obj.run(pytorch_code, ["result_before", "result_inside", "result_after"])


@pytest.mark.skipif(condition=should_skip, reason=skip_reason)
def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result_before = torch.is_autocast_enabled()
        with torch.autocast(device_type='cuda', enabled=False):
            result_inside = torch.is_autocast_enabled()
        result_after = torch.is_autocast_enabled()
        """
    )
    obj.run(pytorch_code, ["result_before", "result_inside", "result_after"])


@pytest.mark.skipif(condition=should_skip, reason=skip_reason)
def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result_L0 = torch.is_autocast_enabled()
        with torch.autocast(device_type='cuda', enabled=True):
            result_L1 = torch.is_autocast_enabled()
            with torch.autocast(device_type='cuda', enabled=False):
                result_L2 = torch.is_autocast_enabled()
            result_back_L1 = torch.is_autocast_enabled()
        result_final = torch.is_autocast_enabled()
        """
    )
    obj.run(
        pytorch_code,
        [
            "result_L0",
            "result_L1",
            "result_L2",
            "result_back_L1",
            "result_final",
        ],
    )
