# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
from apibase import APIBase


class DistributionAPIBase(APIBase):
    """APIBase with custom compare logic for Distribution objects."""

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
        if hasattr(pytorch_result, "batch_shape") and hasattr(
            pytorch_result, "event_shape"
        ):
            assert hasattr(paddle_result, "batch_shape") and hasattr(
                paddle_result, "event_shape"
            ), f"API ({name}): paddle result should be a Distribution object, but got {type(paddle_result)}"

            assert (
                pytorch_result.batch_shape == paddle_result.batch_shape
            ), f"API ({name}): batch_shape mismatch, torch is {pytorch_result.batch_shape}, paddle is {paddle_result.batch_shape}"
            assert (
                pytorch_result.event_shape == paddle_result.event_shape
            ), f"API ({name}): event_shape mismatch, torch is {pytorch_result.event_shape}, paddle is {paddle_result.event_shape}"

            for attr in ("probs", "logits"):
                p_src = getattr(pytorch_result, attr, None)
                d_src = getattr(paddle_result, attr, None)
                if p_src is not None and d_src is not None:
                    np.testing.assert_allclose(
                        p_src.detach().cpu().numpy(),
                        d_src.numpy(),
                        rtol=rtol,
                        atol=atol,
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
