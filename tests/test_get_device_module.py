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

from apibase import APIBase

class DeviceModuleAPIBase(APIBase):
    def compare(
        self,
        name,
        pytorch_result,
        paddle_result,
        check_value=True,
        check_shape=True,
        check_dtype=True,
        check_stop_gradient=False,
        rtol=1.0e-6,
        atol=0.0,
    ):
        assert pytorch_result == paddle_result

obj = DeviceModuleAPIBase("torch.get_device_module")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.get_device_module("cuda")
        result = type(result)
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        check_stop_gradient=False,
    )

def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.get_device_module(torch.device('cuda'))
        result = type(result)
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        check_stop_gradient=False,
    )

def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.get_device_module(torch.device('cuda:0'))
        result = type(result)
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        check_stop_gradient=False,
    )

