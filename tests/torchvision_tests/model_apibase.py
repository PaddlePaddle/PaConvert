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
import paddle
import torch
from apibase import APIBase


class ModelAPIBase(APIBase):
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
        Compare models from PyTorch and PaddlePaddle.
        """
        if isinstance(pytorch_result, torch.nn.Module) and isinstance(
            paddle_result, paddle.nn.Module
        ):
            if "inception_v3" in name:
                pytorch_result.eval()
                paddle_result.eval()
                simple_input = np.random.rand(1, 3, 299, 299).astype(np.float32)
            else:
                simple_input = np.random.rand(1, 3, 224, 224).astype(np.float32)

            pytorch_output = pytorch_result(torch.tensor(simple_input))
            paddle_output = paddle_result(paddle.to_tensor(simple_input))

            if isinstance(pytorch_output, torch.Tensor):
                pytorch_numpy = pytorch_output.detach().cpu().numpy()
            elif isinstance(pytorch_output, tuple):
                pytorch_numpy = np.array(
                    [item.detach().cpu().numpy() for item in pytorch_output]
                )
            else:
                raise ValueError("Unsupported type for pytorch_output")

            if isinstance(paddle_output, paddle.Tensor):
                paddle_numpy = paddle_output.detach().numpy()
            elif isinstance(paddle_output, tuple):
                paddle_numpy = np.array(
                    [item.detach().numpy() for item in paddle_output]
                )
            else:
                raise ValueError("Unsupported type for paddle_output")

            assert (
                pytorch_numpy.shape == paddle_numpy.shape
            ), "API ({}): shape mismatch, torch result shape is {}, paddle result shape is {}".format(
                name, pytorch_numpy.shape, paddle_numpy.shape
            )

        else:
            assert isinstance(
                pytorch_result, type(paddle_result)
            ), "API ({}): model type mismatch, torch model type is {}, paddle model type is {}".format(
                name, type(pytorch_result), type(paddle_result)
            )
