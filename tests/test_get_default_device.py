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

import textwrap

import paddle
import pytest
from test_device import DeviceAPIBase

obj = DeviceAPIBase("torch.get_default_device")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        torch.set_default_device('cpu')
        result = torch.get_default_device()

        # if not set None, will cause test_vander error
        torch.set_default_device(None)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        torch.set_default_device(device=torch.device("cpu:1"))
        result = torch.get_default_device()

        # if not set None, will cause test_vander error
        torch.set_default_device(None)
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="can only run on paddle with CUDA",
)
def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        torch.set_default_device(device=torch.device("cuda:0"))
        result = torch.get_default_device()
        """
    )
    obj.run(pytorch_code, ["result"])
