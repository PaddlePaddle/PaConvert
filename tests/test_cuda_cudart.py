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
from types import BuiltinFunctionType, ModuleType

import paddle
import pytest
from apibase import APIBase


class CudaRtAPIBase(APIBase):
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
        assert isinstance(pytorch_result, ModuleType) and isinstance(
            paddle_result, ModuleType
        )
        for func_name in [
            "cudaGetErrorString",
            "cudaProfilerStart",
            "cudaProfilerStop",
            "cudaHostRegister",
            "cudaHostUnregister",
            "cudaStreamCreate",
            "cudaStreamDestroy",
            "cudaMemGetInfo",
        ]:
            pt_func = getattr(pytorch_result, func_name)
            pd_func = getattr(paddle_result, func_name)
            assert isinstance(pt_func, BuiltinFunctionType), type(pt_func)
            assert isinstance(pd_func, BuiltinFunctionType), type(pd_func)


obj = CudaRtAPIBase("torch.cuda.cudart")


@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="can only run on paddle with CUDA",
)
def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.cuda.cudart()
        """
    )
    obj.run(pytorch_code, ["result"])


obj_base = APIBase("torch.cuda.cudart")


@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="can only run on paddle with CUDA",
)
def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        rt = torch.cuda.cudart()
        result = rt.cudaMemGetInfo(0)
        """
    )
    obj_base.run(pytorch_code, ["result"])
