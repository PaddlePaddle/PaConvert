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

obj = APIBase("torch.nn.SyncBatchNorm")


# All test cases need to run on GPU
@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="can only run on paddle with CUDA",
)
def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch.nn as nn
        import torch
        m = nn.SyncBatchNorm(2)
        input = torch.tensor([[[[0.3, 0.4], [0.3, 0.07]], [[0.83, 0.37], [0.18, 0.93]]]])
        result = m(input)
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="can only run on paddle with CUDA",
)
def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch.nn as nn
        import torch
        m = nn.SyncBatchNorm(2, affine=True)
        input = torch.tensor([[[[0.3, 0.4], [0.3, 0.07]], [[0.83, 0.37], [0.18, 0.93]]]])
        result = m(input)
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
        import torch.nn as nn
        import torch
        m = nn.SyncBatchNorm(2, affine=False)
        input = torch.tensor([[[[0.3, 0.4], [0.3, 0.07]], [[0.83, 0.37], [0.18, 0.93]]]])
        result = m(input)
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="can only run on paddle with CUDA",
)
def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch.nn as nn
        import torch
        m = nn.SyncBatchNorm(2, affine=True, momentum=0.1)
        input = torch.tensor([[[[0.3, 0.4], [0.3, 0.07]], [[0.83, 0.37], [0.18, 0.93]]]])
        result = m(input)
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="can only run on paddle with CUDA",
)
def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch.nn as nn
        import torch
        m = nn.SyncBatchNorm(num_features=2, affine=False, momentum=0.1)
        input = torch.tensor([[[[0.3, 0.4], [0.3, 0.07]], [[0.83, 0.37], [0.18, 0.93]]]])
        result = m(input)
        """
    )
    obj.run(pytorch_code, ["result"])
