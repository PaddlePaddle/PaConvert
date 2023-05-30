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

obj = APIBase("torch.nn.ChannelShuffle")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        x = torch.tensor([[[[0.]],[[0.10000000]],[[0.20000000]],[[0.30000001]],[[0.40000001]],[[0.50000000]]]])
        model = nn.ChannelShuffle(3)
        result = model(x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        x = torch.tensor([[[[0.]],[[0.10000000]],[[0.20000000]],[[0.30000001]],[[0.40000001]],[[0.50000000]]]])
        model = nn.ChannelShuffle(groups=2)
        result = model(x)
        """
    )
    obj.run(pytorch_code, ["result"])
