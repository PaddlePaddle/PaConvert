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
#

import textwrap

import paddle
import pytest
from apibase import APIBase

obj = APIBase("flash_attn.layers.rotary.test_flash_attn_apply_rotary_emb_func")


# Note: FlashAttention only supports Ampere GPUs or newer and CUDA 11.6 or above
@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda()
    or not paddle.device.cuda.get_device_properties(0).major >= 8
    or not float(paddle.version.cuda_version) >= 11.6,
    reason="computational capabilities less 8 or cuda_version less 11.6",
)
def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        # x.shape [1,2,2,4]
        x = torch.tensor([[
            [[3.4742,  0.5466, -0.8008, -0.9079],[3.4742,  0.5466, -0.8008, -0.9079]],
            [[3.4742,  0.5466, -0.8008, -0.9079],[3.4742,  0.5466, -0.8008, -0.9079]]
            ]])
        # cos.shape [1,2,1,4]
        cos = torch.tensor([[
            [[0.4742,  -0.5466, 0.8008, 0.9079]],
            [[0.4742,  -0.5466, 0.8008, 0.9079]]
            ]])
        # sin.shape [1,2,1,4]
        sin = cos
        result = flash_attn.layers.rotary.test_flash_attn_apply_rotary_emb_func(x,cos,sin)
        """
    )
    obj.run(pytorch_code, ["result"])
