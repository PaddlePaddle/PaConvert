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

import paddle
import pytest
from apibase import APIBase

class DevicePropertiesAPIBase(APIBase):
    def compare(
        self,
        name,
        pytorch_result,
        paddle_result,
        check_value=True,
        check_shape=True,
        check_dtype=True,
        check_stop_gradient=False,
        rtol=1.0e-6,
        atol=0.0,
    ):
        assert pytorch_result.name == paddle_result.name
        assert pytorch_result.major == paddle_result.major
        assert pytorch_result.minor == paddle_result.minor
        assert pytorch_result.total_memory == paddle_result.total_memory
        assert pytorch_result.multi_processor_count == paddle_result.multi_processor_count

obj = DevicePropertiesAPIBase("torch.cuda.get_device_properties")


@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="can only run on paddle with CUDA",
)
def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.cuda.get_device_properties()
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
    )
