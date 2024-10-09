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

obj = APIBase("torch.isposinf")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([-float('inf'), float('inf'), 1.2])
        result = torch.isposinf(x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        out = torch.tensor([False, False, False])
        x = torch.tensor([-float('inf'), float('inf'), 1.2])
        result = torch.isposinf(x, out = out)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[-float('inf'), float('inf'), 1.2, 0., 2.5],
                        [-1.35 , -float('inf') ,  0.18, -0.33,  float('inf')],
                        [-float('inf'), float('inf'), 1., 2., 4.]])
        result = torch.isposinf(x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        out = torch.zeros(3, 5, dtype=torch.bool)
        x = torch.tensor([[-float('inf'), float('inf'), 1.2, 0., 2.5],
                        [-1.35 , -float('inf') ,  0.18, -0.33,  float('inf')],
                        [-float('inf'), float('inf'), 1., 2., 4.]])
        result = torch.isposinf(x, out = out)
        """
    )
    obj.run(pytorch_code, ["result"])
