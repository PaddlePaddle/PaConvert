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


class DTypeAPIBase(APIBase):
    """APIBase with custom compare logic for non-Tensor results (e.g., dtype)."""

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
            # Non-Tensor object (e.g., dtype) - compare values directly
            pytorch_str = str(pytorch_result)
            paddle_str = str(paddle_result)
            # Normalize framework prefixes (torch. vs paddle.) for dtype comparison
            pytorch_normalized = pytorch_str.replace("torch.", "").replace(
                "paddle.", ""
            )
            paddle_normalized = paddle_str.replace("torch.", "").replace("paddle.", "")
            assert pytorch_normalized == paddle_normalized, (
                f"API ({name}): value mismatch, "
                f"torch result is {pytorch_result}, paddle result is {paddle_result}"
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


obj = DTypeAPIBase("torch.torch.Tensor")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.torch.Tensor([1, 2, 3])
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.torch.Tensor(3, 4)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.torch.Tensor(3, 4, 5)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.torch.Tensor([1.0, 2.0, 3.0])
        result = a.dtype
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.torch.Tensor(2, 3)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)
