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

obj = APIBase("torch.distributions.exp_family.ExponentialFamily")


@pytest.mark.skip(
    reason="Paddle framework issue: paddle.distribution.ExponentialFamily unpacks tensor argument into batch_shape/event_shape tuples, resulting in different return type than PyTorch"
)
def test_case_6():
    """Expression argument test"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.distributions.exp_family.ExponentialFamily(torch.tensor([1.0, 2.0, 3.0]) + torch.tensor([0.5, 0.5, 0.5]))
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skip(
    reason="Paddle framework issue: paddle.distribution.ExponentialFamily unpacks tensor argument into batch_shape/event_shape tuples, resulting in different return type than PyTorch"
)
def test_case_7():
    """2D tensor test"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[1.4309, 1.2706], [-0.8562, 0.9796]])
        result = torch.distributions.exp_family.ExponentialFamily(a)
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skip(
    reason="Paddle framework issue: paddle.distribution.ExponentialFamily unpacks tensor argument into batch_shape/event_shape tuples, resulting in different return type than PyTorch"
)
def test_case_8():
    """3D tensor test"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]])
        result = torch.distributions.exp_family.ExponentialFamily(a)
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skip(
    reason="Paddle framework issue: paddle.distribution.ExponentialFamily unpacks tensor argument into batch_shape/event_shape tuples, resulting in different return type than PyTorch"
)
def test_case_9():
    """float64 dtype test"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([1.4309, 1.2706], dtype=torch.float64)
        result = torch.distributions.exp_family.ExponentialFamily(a)
        """
    )
    obj.run(pytorch_code, ["result"])
