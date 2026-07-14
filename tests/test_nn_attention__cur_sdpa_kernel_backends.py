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


class SDPBackendListAPIBase(APIBase):
    """APIBase with custom compare logic for lists of SDPBackend enum values."""

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
        if isinstance(pytorch_result, (list, tuple)) and all(
            not hasattr(x, "requires_grad") for x in pytorch_result
        ):
            assert isinstance(paddle_result, (list, tuple)), (
                f"API ({name}): paddle result should be a list/tuple, "
                f"got {type(paddle_result)}"
            )
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


obj = SDPBackendListAPIBase("torch.nn.attention._cur_sdpa_kernel_backends")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.nn.attention._cur_sdpa_kernel_backends()
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        backends = torch.nn.attention._cur_sdpa_kernel_backends()
        result = all(isinstance(b, torch.nn.attention.SDPBackend) for b in backends)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch

        with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
            backends = torch.nn.attention._cur_sdpa_kernel_backends()
            result = len(backends)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch.nn.attention as attn
        result = attn._cur_sdpa_kernel_backends()
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)
