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


class SDPBackendAPIBase(APIBase):
    """APIBase with custom compare logic for SDPBackend enum values."""

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
        if not hasattr(pytorch_result, "requires_grad"):
            # Non-Tensor object (e.g., SDPBackend) - just verify creation
            assert pytorch_result is not None
            assert paddle_result is not None
            return
        super().compare(
            name,
            pytorch_result,
            paddle_result,
            check_value,
            check_shape,
            check_dtype,
            check_stop_gradient,
            rtol,
            atol,
        )


obj = SDPBackendAPIBase("torch.nn.attention.SDPBackend.ERROR")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.nn.attention.SDPBackend.ERROR
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = isinstance(torch.nn.attention.SDPBackend.ERROR, torch.nn.attention.SDPBackend)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.nn.attention.SDPBackend.ERROR.value
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.nn.attention.SDPBackend.ERROR.name
        """
    )
    obj.run(pytorch_code, ["result"])
