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


class cudaSetDeviceAPI(APIBase):
    def compare(
        self,
        name,
        pytorch_result,
        paddle_result,
        check_value=True,
        check_dtype=True,
        check_stop_gradient=True,
        rtol=1.0e-6,
        atol=0.0,
    ):

        assert pytorch_result == paddle_result.get_device_id()


obj = cudaSetDeviceAPI("torch.cuda.set_device")


@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="can only run on paddle with CUDA",
)
def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        torch.cuda.set_device(0)
        result = torch.cuda.current_device()
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
        import torch
        torch.cuda.set_device(device=1)
        result = torch.cuda.current_device()
        """
    )
    obj.run(pytorch_code, ["result"])


# NOTE: why not run?
# paddle.device.set_device should support CUDAPlace/CPUPlace, but not supported currently.
@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="can only run on paddle with CUDA",
)
def _test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        t = torch.tensor([1,2,3]).cuda()
        result = torch.cuda.set_device(torch.device("cuda:0"))
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="can only run on paddle with CUDA",
)
def _test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        t = torch.tensor([1,2,3]).cuda()
        result = torch.cuda.set_device(device=torch.device("cuda:0"))
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
        import torch
        num = 1
        torch.cuda.set_device(device=f"cuda:{num}")
        result = torch.cuda.current_device()
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="can only run on paddle with CUDA",
)
def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        torch.cuda.set_device(device=0 if 2 > 1 else 1)
        result = torch.cuda.current_device()
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="can only run on paddle with CUDA",
)
def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        torch.cuda.set_device(device="cuda:0" if 2 > 1 else "cuda:1")
        result = torch.cuda.current_device()
        """
    )
    obj.run(pytorch_code, ["result"])
