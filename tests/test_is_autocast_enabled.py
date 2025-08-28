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

from apibase import APIBase

obj = APIBase("torch.is_autocast_enabled")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        if torch.cuda.is_available():
            result = torch.is_autocast_enabled()
        else:
            result = False
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        if torch.cuda.is_available():
            result_before = torch.is_autocast_enabled()
            with torch.autocast(device_type='cuda', enabled=True):
                result_inside = torch.is_autocast_enabled()
            result_after = torch.is_autocast_enabled()
        else:
            result_before = result_inside = result_after = False
        """
    )
    obj.run(pytorch_code, ["result_before", "result_inside", "result_after"])


def test_case_3():
    # 对应 test_amp_autocast_false: 检查 GPU autocast(enabled=False) 的行为
    pytorch_code = textwrap.dedent(
        """
        import torch
        if torch.cuda.is_available():
            result_before = torch.is_autocast_enabled()
            with torch.autocast(device_type='cuda', enabled=False):
                result_inside = torch.is_autocast_enabled()
            result_after = torch.is_autocast_enabled()
        else:
            result_before = result_inside = result_after = False
        """
    )
    obj.run(pytorch_code, ["result_before", "result_inside", "result_after"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        if torch.cuda.is_available():
            result_L0 = torch.is_autocast_enabled()

            with torch.autocast(device_type='cuda', enabled=True):
                result_L1 = torch.is_autocast_enabled()

                with torch.autocast(device_type='cuda', enabled=False):
                    result_L2 = torch.is_autocast_enabled()

                result_back_L1 = torch.is_autocast_enabled()

            result_final = torch.is_autocast_enabled()
        else:
            result_L0 = result_L1 = result_L2 = result_back_L1 = result_final = False
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
