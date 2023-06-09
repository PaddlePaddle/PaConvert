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
        unsupport=False,
        reason=None,
    ) -> None:
        """
        args:
            pytorch_code: pytorch code to execute
            compared_tensor_names: the list of variant name to be compared
            expect_paddle_code: the string of expect paddle code
            check_value: If false, the value will not be checked
            unsupport: If true, conversion is not supported
            reason: the reason why it is not supported
        """
        if unsupport:
            assert (
                reason is not None
            ), "Please explain the reason why it is not supported"
            paddle_code = self.convert(pytorch_code)
            assert ">>>" in paddle_code
            return
        if compared_tensor_names:
            loc = locals()
            exec(pytorch_code)
            pytorch_result = [loc[name] for name in compared_tensor_names]

            paddle_code = self.convert(pytorch_code)
            exec(paddle_code)
            paddle_result = [loc[name] for name in compared_tensor_names]
            for i in range(len(compared_tensor_names)):
                self.compare(
                    self.pytorch_api, pytorch_result[i], paddle_result[i], check_value
                )

        if expect_paddle_code:
            convert_paddle_code = self.convert(pytorch_code)
            assert (
                convert_paddle_code == expect_paddle_code
            ), "[{}]: get unexpected code".format(self.pytorch_api)

    def compare(self, name, pytorch_result, paddle_result, check_value=True):
        """
        compare tensors' data, shape, requires_grad, dtype
        args:
            name: pytorch api name
            pytorch_result: pytorch Tensor
            paddle_result: paddle Tensor
            check_value: If false, the value will not be checked
        """
        if isinstance(pytorch_result, (tuple, list)):
            assert isinstance(
                paddle_result, (tuple, list)
            ), "paddle result shoule be list/tuple"
            assert len(pytorch_result) == len(
                paddle_result
            ), "paddle result have different length with pytorch"
            for i in range(len(pytorch_result)):
                self.compare(self.pytorch_api, pytorch_result[i], paddle_result[i])
            return

        if isinstance(pytorch_result, (bool, np.number, int, str)):
            assert isinstance(
                paddle_result, (bool, np.number, int, str)
            ), "paddle result shoule be bool/np.number/int/str"
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
            pytorch_numpy, paddle_numpy = pytorch_result.numpy(), paddle_result.numpy(
                False
            )

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
        assert (
            pytorch_numpy.dtype == paddle_numpy.dtype
        ), "API ({}): dtype mismatch, torch dtype is {}, paddle dtype is {}".format(
            name, pytorch_numpy.dtype, paddle_numpy.dtype
        )
        if check_value:
            assert np.allclose(
                pytorch_numpy, paddle_numpy
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

        coverter = Converter(log_dir="disable")
        coverter.run(pytorch_code_path, paddle_code_path)

        with open(paddle_code_path, "r", encoding="UTF-8") as f:
            code = f.read()
        return code
