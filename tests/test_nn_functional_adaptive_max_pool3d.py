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

import numpy as np
from apibase import APIBase


class MaxPoolAPI(APIBase):
    def compare(self, name, pytorch_result, paddle_result, check_value=True):
        if isinstance(pytorch_result, (tuple, list)):
            for i in range(len(pytorch_result)):
                self.compare(self.pytorch_api, pytorch_result[i], paddle_result[i])
            return

        pytorch_numpy, paddle_numpy = pytorch_result.numpy(), paddle_result.numpy()
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
        assert np.allclose(
            pytorch_numpy, paddle_numpy
        ), "API ({}): paddle result has diff with pytorch result".format(name)


obj = MaxPoolAPI("torch.nn.functional.adaptive_max_pool3d")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        x = torch.tensor([[[[[-1.1494, -1.3829],
                            [ 0.4995, -1.3094]],
                            [[ 1.0015,  1.4919],
                            [-1.5187,  0.0235]]]]])
        result = nn.functional.adaptive_max_pool3d(x, (1, 1, 1))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        x = torch.tensor([[[[[-1.1494, -1.3829],
                            [ 0.4995, -1.3094]],
                            [[ 1.0015,  1.4919],
                            [-1.5187,  0.0235]]]]])
        result = nn.functional.adaptive_max_pool3d(x, output_size=1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        x = torch.tensor([[[[[-1.1494, -1.3829],
                            [ 0.4995, -1.3094]],
                            [[ 1.0015,  1.4919],
                            [-1.5187,  0.0235]]]]])
        result = nn.functional.adaptive_max_pool3d(x, 1, False)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        x = torch.tensor([[[[[-1.1494, -1.3829],
                            [ 0.4995, -1.3094]],
                            [[ 1.0015,  1.4919],
                            [-1.5187,  0.0235]]]]])
        result, indices = nn.functional.adaptive_max_pool3d(x, 1, True)
        """
    )
    obj.run(pytorch_code, ["result", "indices"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        x = torch.tensor([[[[[-1.1494, -1.3829],
                            [ 0.4995, -1.3094]],
                            [[ 1.0015,  1.4919],
                            [-1.5187,  0.0235]]]]])
        result = nn.functional.adaptive_max_pool3d(input=x, output_size=1, return_indices=False)
        """
    )
    obj.run(pytorch_code, ["result"])
