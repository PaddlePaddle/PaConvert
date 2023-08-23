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

obj = APIBase("torch.nn.functional.group_norm")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.tensor([[[[-1.2392, -0.1310, -0.6679,  0.5476],
                            [ 1.1738, -1.7384, -0.7733,  0.3261],
                            [-0.0926, -1.0448, -1.2557, -1.5503],
                            [ 0.6402,  0.9072,  0.6780, -1.9885]],

                            [[ 0.0639, -1.1592,  1.4242, -0.4641],
                            [-0.1920,  0.1826,  1.9217, -0.4359],
                            [ 1.1926, -0.0247,  0.4744, -1.0216],
                            [-0.0360, -1.1656,  0.3661, -1.8147]]]])
        result = F.group_norm(x, 3)
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        unsupport=True,
        reason="paddle does not support this function temporarily",
    )


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.tensor([[[[-1.2392, -0.1310, -0.6679,  0.5476],
                            [ 1.1738, -1.7384, -0.7733,  0.3261],
                            [-0.0926, -1.0448, -1.2557, -1.5503],
                            [ 0.6402,  0.9072,  0.6780, -1.9885]],

                            [[ 0.0639, -1.1592,  1.4242, -0.4641],
                            [-0.1920,  0.1826,  1.9217, -0.4359],
                            [ 1.1926, -0.0247,  0.4744, -1.0216],
                            [-0.0360, -1.1656,  0.3661, -1.8147]]]])
        result = F.group_norm(x, 3)
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        unsupport=True,
        reason="paddle does not support this function temporarily",
    )


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.tensor([[[[-1.2392, -0.1310, -0.6679,  0.5476],
                            [ 1.1738, -1.7384, -0.7733,  0.3261],
                            [-0.0926, -1.0448, -1.2557, -1.5503],
                            [ 0.6402,  0.9072,  0.6780, -1.9885]],

                            [[ 0.0639, -1.1592,  1.4242, -0.4641],
                            [-0.1920,  0.1826,  1.9217, -0.4359],
                            [ 1.1926, -0.0247,  0.4744, -1.0216],
                            [-0.0360, -1.1656,  0.3661, -1.8147]]]])
        result = F.group_norm(x, 3, eps=1e-5)
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        unsupport=True,
        reason="paddle does not support this function temporarily",
    )
