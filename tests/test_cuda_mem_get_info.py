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

obj = APIBase("torch.cuda.mem_get_info")


@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="can only run on paddle with CUDA",
)
def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.cuda.mem_get_info()
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        reason="paddle does not support this function temporarily",
    )


@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="can only run on paddle with CUDA",
)
def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        t = torch.tensor([1,2,3]).cuda()
        result = torch.cuda.mem_get_info()
        """
    )
    obj.run(
        pytorch_code,
        [],
        reason="paddle does not support this function temporarily",
    )


@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="can only run on paddle with CUDA",
)
def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        t = torch.tensor([1,2,3]).cuda()
        result = torch.cuda.mem_get_info(0)
        """
    )
    obj.run(
        pytorch_code,
        [],
        reason="paddle does not support this function temporarily",
    )


@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="can only run on paddle with CUDA",
)
def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        t = torch.tensor([1,2,3]).cuda()
        result = torch.cuda.mem_get_info(device=0)
        """
    )
    obj.run(
        pytorch_code,
        [],
        reason="paddle does not support this function temporarily",
    )


@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="can only run on paddle with CUDA",
)
def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        t = torch.tensor([1,2,3]).cuda()
        result = torch.cuda.mem_get_info(torch.device("cuda:0"))
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        reason="paddle does not support this function temporarily",
    )


@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="can only run on paddle with CUDA",
)
def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        t = torch.tensor([1,2,3]).cuda()
        result = torch.cuda.mem_get_info(device=torch.device("cuda:0"))
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        reason="paddle does not support this function temporarily",
    )


class CudaMemGetInfoAPIBase(APIBase):
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
        assert isinstance(pytorch_result, tuple) and len(pytorch_result) == 2
        assert isinstance(paddle_result, tuple) and len(paddle_result) == 2


cuda_mem_get_info_obj = CudaMemGetInfoAPIBase("torch.cuda.mem_get_info")


@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="can only run on paddle with CUDA",
)
def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.cuda.mem_get_info()
        """
    )
    cuda_mem_get_info_obj.run(pytorch_code, ["result"])
