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

obj = APIBase("flash_attn.flash_attn_interface.flash_attn_unpadded_func")


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
        from flash_attn.flash_attn_interface import flash_attn_varlen_func as flash_attn_unpadded_func
        # q.shape [2,2,4]
        q = torch.ones([8,8,8],dtype=torch.float16).cuda()
        cu_seqlens_q = torch.ones([8],dtype=torch.int32).cuda()
        result = flash_attn_unpadded_func(q,q,q,cu_seqlens_q,cu_seqlens_q,4,4,0.25)
        """
    )
    obj.run(pytorch_code, ["result"])
