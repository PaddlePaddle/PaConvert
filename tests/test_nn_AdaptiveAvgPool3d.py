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

obj = APIBase("torch.nn.AdaptiveAvgPool3d")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        x = torch.tensor([[[[[-1.1494, -1.3829],
                            [ 0.4995, -1.3094]],
                            [[ 1.0015,  1.4919],
                            [-1.5187,  0.0235]]]]])
        model = nn.AdaptiveAvgPool3d(1)
        result = model(x)
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
        model = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        result = model(x)
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
        model = nn.AdaptiveAvgPool3d(output_size=1)
        result = model(x)
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
        model = nn.AdaptiveAvgPool3d(output_size=(None, None, 1))
        result = model(x)
        """
    )
    obj.run(pytorch_code, ["result"])
