# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from PIL import Image


class ImageAPIBase(APIBase):
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
        """
        Compare PIL Images for equality.
        """
        if isinstance(pytorch_result, Image.Image) and isinstance(
            paddle_result, Image.Image
        ):
            pytorch_array = np.array(pytorch_result)
            paddle_array = np.array(paddle_result)

            assert (
                pytorch_array.shape == paddle_array.shape
            ), "API ({}): shape mismatch, torch shape is {}, paddle shape is {}".format(
                name, pytorch_array.shape, paddle_array.shape
            )

            if check_value:
                np.testing.assert_allclose(
                    pytorch_array, paddle_array, rtol=rtol, atol=atol
                ), "API ({}): paddle result has diff with pytorch result".format(name)
            return

        super().compare(
            name,
            pytorch_result,
            paddle_result,
            check_value,
            check_dtype,
            check_stop_gradient,
            rtol,
            atol,
        )
