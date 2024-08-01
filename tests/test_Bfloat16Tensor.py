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
import numpy as np

from apibase import APIBase


class Bfloat16TensorAPIBase(APIBase):
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
        """
        compare tensors' data, shape, requires_grad, dtype
        args:
            name: pytorch api name
            pytorch_result: pytorch Tensor
            paddle_result: paddle Tensor
            check_value: If false, the value will not be checked
            check_dtype: If false, the dtype will not be checked
            check_stop_gradient: If false, the stop gradient will not be checked
        """
        if isinstance(pytorch_result, dict):
            assert isinstance(paddle_result, dict), "paddle result should be dict"
            assert len(pytorch_result) == len(
                paddle_result
            ), "paddle result have different length with pytorch"
            pytorch_result_k = [k for k in pytorch_result.keys()]
            pytorch_result_v = [v for v in pytorch_result.values()]
            paddle_result_k = [k for k in paddle_result.keys()]
            paddle_result_v = [v for v in paddle_result.values()]
            self.compare(
                self.pytorch_api,
                pytorch_result_k,
                paddle_result_k,
                check_value,
                check_dtype,
                check_stop_gradient,
                rtol,
                atol,
            )
            self.compare(
                self.pytorch_api,
                pytorch_result_v,
                paddle_result_v,
                check_value,
                check_dtype,
                check_stop_gradient,
                rtol,
                atol,
            )
            return

        if isinstance(pytorch_result, (tuple, list)):
            assert isinstance(
                paddle_result, (tuple, list)
            ), "paddle result should be list/tuple"
            assert len(pytorch_result) == len(
                paddle_result
            ), "paddle result have different length with pytorch"
            for i in range(len(pytorch_result)):
                self.compare(
                    self.pytorch_api,
                    pytorch_result[i],
                    paddle_result[i],
                    check_value,
                    check_dtype,
                    check_stop_gradient,
                    rtol,
                    atol,
                )
            return

        if isinstance(pytorch_result, (bool, np.number, int, float, str, type(None))):
            assert type(paddle_result) == type(
                pytorch_result
            ), "paddle result's type [{}] should be the same with pytorch's type [{}]".format(
                type(paddle_result), type(pytorch_result)
            )
            if check_value:
                assert (
                    pytorch_result == paddle_result
                ), "API ({}): pytorch result is {}, but paddle result is {}".format(
                    name, pytorch_result, paddle_result
                )
            return

        if pytorch_result.requires_grad:
            pytorch_numpy, paddle_numpy = (
                pytorch_result.detach().numpy(),
                paddle_result.numpy(False),
            )
        elif pytorch_result.is_conj():
            pytorch_numpy, paddle_numpy = (
                pytorch_result.resolve_conj().numpy(),
                paddle_result.numpy(False),
            )
        else:
            (
                pytorch_numpy,
                paddle_numpy,
            ) = pytorch_result.float().cpu().numpy(), paddle_result.astype('float32').numpy(False)

        if check_stop_gradient:
            assert (
                pytorch_result.requires_grad != paddle_result.stop_gradient
            ), "API ({}): requires grad mismatch, torch tensor's requires_grad is {}, paddle tensor's stop_gradient is {}".format(
                name, pytorch_result.requires_grad, paddle_result.stop_gradient
            )

        assert (
            pytorch_numpy.shape == paddle_numpy.shape
        ), "API ({}): shape mismatch, torch shape is {}, paddle shape is {}".format(
            name, pytorch_numpy.shape, paddle_numpy.shape
        )
        if check_dtype:
            assert (
                pytorch_numpy.dtype == paddle_numpy.dtype
            ), "API ({}): dtype mismatch, torch dtype is {}, paddle dtype is {}".format(
                name, pytorch_numpy.dtype, paddle_numpy.dtype
            )
        if check_value:
            assert np.allclose(
                pytorch_numpy, paddle_numpy, rtol=rtol, atol=atol
            ), "API ({}): paddle result has diff with pytorch result".format(name)

obj = Bfloat16TensorAPIBase("torch.BFloat16Tensor")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.BFloat16Tensor([1.5, 2, 3])
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.BFloat16Tensor()
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.BFloat16Tensor(2)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.BFloat16Tensor(2,3)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)
