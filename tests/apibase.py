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

import difflib
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(__file__) + "/..")

from paconvert.converter import Converter


class APIBase(object):
    def __init__(self, pytorch_api) -> None:
        """
        args:
            pytorch_api: The corresponding pytorch api
        """
        self.pytorch_api = pytorch_api

    def run(
        self,
        pytorch_code,
        compared_tensor_names=None,
        expect_paddle_code=None,
        check_value=True,
        check_dtype=True,
        check_stop_gradient=True,
        rtol=1.0e-6,
        atol=0.0,
        unsupport=False,
        reason=None,
    ) -> None:
        """
        args:
            pytorch_code: pytorch code to execute
            compared_tensor_names: the list of variant name to be compared
            expect_paddle_code: the string of expect paddle code
            check_value: If false, the value will not be checked
            check_dtype: If false, the dtype will not be checked
            check_stop_gradient: If false, the stop gradient will not be checked
            unsupport: If true, conversion is not supported
            reason: the reason why it is not supported
        """
        if unsupport:
            assert (
                reason is not None
            ), "Please explain the reason why it is not supported"
            paddle_code = self.convert(pytorch_code)
            assert ">>>>>>" in paddle_code
            return
        if compared_tensor_names:
            loc = locals()
            exec(pytorch_code, locals())
            pytorch_result = [loc[name] for name in compared_tensor_names]

            paddle_code = self.convert(pytorch_code)
            exec(paddle_code, locals())
            paddle_result = [loc[name] for name in compared_tensor_names]
            for i in range(len(compared_tensor_names)):
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
        if expect_paddle_code:
            converted_code = self.convert(pytorch_code).strip()
            expected_code = expect_paddle_code.strip()
            if converted_code != expected_code:
                diff = difflib.unified_diff(
                    expected_code.splitlines(),
                    converted_code.splitlines(),
                    fromfile="expected",
                    tofile="converted",
                    lineterm="",
                )
                diff_text = "\n".join(diff)
                error_msg = (
                    f"[{self.pytorch_api}] Code conversion result differs from expectation:\n"
                    f"{'-'*50}\n"
                    f"Diff comparison:\n"
                    f"{diff_text}\n"
                    f"{'-'*50}"
                )
                assert converted_code == expected_code, error_msg

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
            ) = pytorch_result.cpu().numpy(), paddle_result.numpy(False)

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

    def convert(self, pytorch_code):
        """
        convert pytorch code to paddle code.
        args:
            pytorch_code: pytorch code to be converted.
        return:
            paddle code.
        """
        if not os.path.exists(os.getcwd() + "/test_project"):
            os.makedirs(os.getcwd() + "/test_project")

        pytorch_code_path = os.getcwd() + "/test_project/pytorch_temp.py"
        paddle_code_path = os.getcwd() + "/test_project/paddle_temp.py"
        with open(pytorch_code_path, "w", encoding="UTF-8") as f:
            f.write(pytorch_code)

        converter = Converter(log_dir="disable")
        converter.run(pytorch_code_path, paddle_code_path)

        with open(paddle_code_path, "r", encoding="UTF-8") as f:
            code = f.read()
        return code
