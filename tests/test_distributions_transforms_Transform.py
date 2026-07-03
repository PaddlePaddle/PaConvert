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


class TransformAPIBase(APIBase):
    """APIBase with custom compare logic for Transform objects.

    Transform objects are not Tensors and cannot be compared using the
    default Tensor-based compare logic. This verifies that the Transform
    was created successfully on both sides.
    """

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
        if hasattr(pytorch_result, "bijective") or hasattr(
            pytorch_result, "_is_injective"
        ):
            assert paddle_result is not None, f"API ({name}): paddle result is None"
            assert hasattr(paddle_result, "_is_injective") or hasattr(
                paddle_result, "bijective"
            ), (
                f"API ({name}): paddle result should be a Transform object, "
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


obj = TransformAPIBase("torch.distributions.transforms.Transform")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.distributions.transforms.Transform()
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.distributions.transforms.Transform(cache_size=0)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.distributions.transforms.Transform(1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.distributions.transforms.Transform(cache_size=1)
        """
    )
    obj.run(pytorch_code, ["result"])
