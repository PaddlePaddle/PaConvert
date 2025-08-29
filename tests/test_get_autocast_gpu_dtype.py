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
from apibase import APIBase

should_skip = not paddle.device.is_compiled_with_cuda()
skip_reason = "AMP test can only run with CUDA."


class GetAutocastGpuTypeAPIBase(APIBase):
    def compare(
        self,
        name,
        pytorch_result,
        paddle_result,
        check_value=True,
        check_shape=True,
        check_dtype=True,
        check_stop_gradient=True,
        rtol=1.0e-6,
        atol=0.0,
    ):
        pytorch_dtype_str = str(pytorch_result).removeprefix("torch.")
        assert pytorch_dtype_str == paddle_result


obj = GetAutocastGpuTypeAPIBase("torch.get_autocast_gpu_dtype")


@pytest.mark.skipif(condition=should_skip, reason=skip_reason)
def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result_gpu = torch.get_autocast_gpu_dtype()
        """
    )
    obj.run(pytorch_code, ["result_gpu"])


@pytest.mark.skipif(condition=should_skip, reason=skip_reason)
def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result_before = torch.get_autocast_gpu_dtype()
        with torch.autocast("cuda",dtype=torch.bfloat16):
            result_inside = torch.get_autocast_gpu_dtype()
        result_after = torch.get_autocast_gpu_dtype()
        """
    )
    obj.run(pytorch_code, ["result_before", "result_inside", "result_after"])


@pytest.mark.skipif(condition=should_skip, reason=skip_reason)
def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result_gpu_level0 = torch.get_autocast_gpu_dtype()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            result_gpu_level1 = torch.get_autocast_gpu_dtype()
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                result_gpu_level2 = torch.get_autocast_gpu_dtype()
            result_gpu_back_to_level1 = torch.get_autocast_gpu_dtype()
        result_gpu_final = torch.get_autocast_gpu_dtype()
        """
    )
    obj.run(
        pytorch_code,
        [
            "result_gpu_level0",
            "result_gpu_level1",
            "result_gpu_level2",
            "result_gpu_back_to_level1",
            "result_gpu_final",
        ],
    )
