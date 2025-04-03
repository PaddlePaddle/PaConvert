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

from test_device import DeviceAPIBase

obj = DeviceAPIBase("torch.cpu.set_device")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        torch.cpu.set_device(device='cpu:0')
        result = torch.cpu.current_device()
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        device = 'cpu'
        torch.cpu.set_device(device=device)
        result = torch.cpu.current_device()
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        device = 'cpu:1'
        torch.cpu.set_device(device=device)
        result = torch.cpu.current_device()
        """
    )
    obj.run(pytorch_code, ["result"])
