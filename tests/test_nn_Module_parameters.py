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

obj = APIBase("torch.nn.Module.parameters")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        model= nn.ReLU()
        list = model.parameters()
        result = []
        for i in list:
            result.append(i)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        model= nn.Conv2d(1, 20, 5)
        list = model.parameters()
        result = []
        for i in list:
            result.append(i)
        weight, bias = result[0], result[1]
        """
    )
    obj.run(pytorch_code, ["weight", "bias"], check_value=False)


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch.nn as nn
        import torch.nn.functional as F

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)

            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))
        model = Model()
        list = model.parameters()
        result = []
        for i in list:
            result.append(i)
        weight0, bias0, weight1, bias1 = result
        """
    )
    obj.run(pytorch_code, ["weight0", "bias0", "weight1", "bias1"], check_value=False)
