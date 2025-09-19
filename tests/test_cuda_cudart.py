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

obj = APIBase("torch.cuda.cudart")


class CudaRtModuleAPIBase(APIBase):
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


cuda_rt_module_obj = CudaRtModuleAPIBase("torch.cuda.cudart")


class CudaIsInitializedAPIBase(APIBase):
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
        assert pytorch_result is True and paddle_result is True


cuda_is_initialized_obj = CudaIsInitializedAPIBase("torch.cuda.is_initialized")


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


class CudaCheckErrorAPIBase(APIBase):
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
        for pt_res, pd_res in zip(pytorch_result, paddle_result):
            assert pt_res == pd_res, f"{pt_res} != {pd_res}"


cuda_check_error_obj = CudaCheckErrorAPIBase("torch.cuda.check_error")


@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="can only run on paddle with CUDA",
)
def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.ones(2, 2).cuda()
        result = torch.cuda.is_initialized()
        """
    )
    cuda_is_initialized_obj.run(pytorch_code, ["result"])


@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="can only run on paddle with CUDA",
)
def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.cuda.cudart()
        """
    )
    cuda_rt_module_obj.run(pytorch_code, ["result"])


@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="can only run on paddle with CUDA",
)
def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.cuda.mem_get_info()
        """
    )
    cuda_mem_get_info_obj.run(pytorch_code, ["result"])


@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="can only run on paddle with CUDA",
)
def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result0 = ""
        result1 = ""
        result2 = ""
        try:
            torch.cuda.check_error(0)
        except RuntimeError as e:
            result0 = str(e)
        try:
            torch.cuda.check_error(1)
        except RuntimeError as e:
            result1 = str(e)
        try:
            torch.cuda.check_error(2)
        except RuntimeError as e:
            result2 = str(e)
        """
    )
    cuda_check_error_obj.run(pytorch_code, ["result0", "result1", "result2"])


@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="can only run on paddle with CUDA",
)
def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        rt = torch.cuda.cudart()
        result = rt.cudaMemGetInfo(0)
        """
    )
    cuda_mem_get_info_obj.run(pytorch_code, ["result"])
