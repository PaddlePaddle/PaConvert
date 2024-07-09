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

obj = APIBase("flash_attn.ops.rms_norm.rms_norm")


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
        import torch
        from flash_attn.ops.rms_norm import rms_norm
        # x.shape [2,1,8]
        x = torch.tensor([
            [[0.4742,  3.5466, -4.8008, -8.9079, 0.4742,  9.5466, -8.8008, -6.9079]],
            [[3.4742,  0.5466, -0.8008, -0.9079, 3.4742,  0.5466, -0.8008, -0.9079]]
            ]).cuda()
        weight = torch.ones(8).cuda()
        result = rms_norm(x, weight,1e-6)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from flash_attn.ops.rms_norm import rms_norm
        # x.shape [2,8]
        x = torch.tensor([
            [0.4742,  3.5466, -4.8008, -8.9079, 0.4742,  9.5466, -8.8008, -6.9079],
            [3.4742,  0.5466, -0.8008, -0.9079, 3.4742,  0.5466, -0.8008, -0.9079]
            ]).cuda()
        weight = torch.ones(8).cuda()
        result = rms_norm(x, weight,1e-6)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from flash_attn.ops.rms_norm import rms_norm
        # x.shape [2,2,2,8]
        x = torch.tensor([
            [[[0.4742,  3.5466, -4.8008, -8.9079, 0.4742,  9.5466, -8.8008, -6.9079],[0.4742,  3.5466, -4.8008, -8.9079, 0.4742,  9.5466, -8.8008, -6.9079]],
            [[3.4742,  0.5466, -0.8008, -0.9079, 3.4742,  0.5466, -0.8008, -0.9079],[3.4742,  0.5466, -0.8008, -0.9079, 3.4742,  0.5466, -0.8008, -0.9079]]],
            [[[0.4742,  3.5466, -4.8008, -8.9079, 0.4742,  9.5466, -8.8008, -6.9079],[0.4742,  3.5466, -4.8008, -8.9079, 0.4742,  9.5466, -8.8008, -6.9079]],
            [[3.4742,  0.5466, -0.8008, -0.9079, 3.4742,  0.5466, -0.8008, -0.9079],[3.4742,  0.5466, -0.8008, -0.9079, 3.4742,  0.5466, -0.8008, -0.9079]]]
            ]).cuda()
        weight = torch.ones(8).cuda()
        result = rms_norm(x, weight,1e-6)
        """
    )
    obj.run(pytorch_code, ["result"])
