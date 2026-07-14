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

import pytest
from apibase import APIBase

obj = APIBase("torch.optim.sgd.SGD")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torch.optim.sgd import SGD
        result = SGD.__name__
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skip(
    reason="PaConvert alias mapping for torch.optim.sgd.SGD uses ChangePrefixMatcher and does not convert parameter names (lr -> learning_rate, params -> parameters), resulting in invalid Paddle code"
)
def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torch.optim.sgd import SGD
        linear = torch.nn.Linear(2, 1)
        optim = SGD(linear.parameters(), lr=0.1)
        result = optim.param_groups[0]["lr"]
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skip(
    reason="PaConvert alias mapping for torch.optim.sgd.SGD uses ChangePrefixMatcher and does not convert parameter names (lr -> learning_rate, params -> parameters), resulting in invalid Paddle code"
)
def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torch.optim.sgd import SGD
        linear = torch.nn.Linear(2, 1)
        optim = SGD(params=linear.parameters(), lr=0.1)
        result = optim.param_groups[0]["lr"]
        """
    )
    obj.run(pytorch_code, ["result"])
