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

obj = APIBase("torch.nn.functional.scaled_dot_product_attention")


@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda()
    or not paddle.device.cuda.get_device_properties(0).major >= 8,
    reason="computational capabilities less 8",
)
def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        query = torch.ones(32, 8, 128, 64, dtype=torch.float16, device="cuda")
        key = torch.ones(32, 8, 128, 64, dtype=torch.float16, device="cuda")
        value = torch.ones(32, 8, 128, 64, dtype=torch.float16, device="cuda")
        result = torch.nn.functional.scaled_dot_product_attention(query,key,value)
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda()
    or not paddle.device.cuda.get_device_properties(0).major >= 8,
    reason="computational capabilities less 8",
)
def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        query = torch.rand(32, 8, 128, 64, dtype=torch.float16, device="cuda")
        key = torch.rand(32, 8, 128, 64, dtype=torch.float16, device="cuda")
        value = torch.rand(32, 8, 128, 64, dtype=torch.float16, device="cuda")
        result = torch.nn.functional.scaled_dot_product_attention(query,key,value)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda()
    or not paddle.device.cuda.get_device_properties(0).major >= 8,
    reason="computational capabilities less 8",
)
def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import math
        query = torch.ones(32, 8, 128, 64, dtype=torch.float16, device="cuda")
        key = torch.ones(32, 8, 128, 64, dtype=torch.float16, device="cuda")
        value = torch.ones(32, 8, 128, 64, dtype=torch.float16, device="cuda")
        result = torch.nn.functional.scaled_dot_product_attention(query,key,value,scale=8)
        """
    )
    obj.run(pytorch_code, ["result"])
