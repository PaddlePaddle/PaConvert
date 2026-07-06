# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

obj = APIBase("torch.nn.modules.loss.MultiMarginLoss")


def test_case_1():
    """default"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input = torch.tensor([[0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 1.0, 1.1, 1.2]])
        target = torch.tensor([0, 2, 3], dtype=torch.long)
        loss = torch.nn.modules.loss.MultiMarginLoss()
        result = loss(input, target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    """p=2"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input = torch.tensor([[0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 1.0, 1.1, 1.2]])
        target = torch.tensor([0, 2, 3], dtype=torch.long)
        loss = torch.nn.modules.loss.MultiMarginLoss(p=2)
        result = loss(input, target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    """margin=1.0"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input = torch.tensor([[0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 1.0, 1.1, 1.2]])
        target = torch.tensor([0, 2, 3], dtype=torch.long)
        loss = torch.nn.modules.loss.MultiMarginLoss(margin=1.0)
        result = loss(input, target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    """weight specified"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input = torch.tensor([[0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 1.0, 1.1, 1.2]])
        target = torch.tensor([0, 2, 3], dtype=torch.long)
        weight = torch.tensor([0.5, 1.0, 1.5, 2.0])
        loss = torch.nn.modules.loss.MultiMarginLoss(weight=weight)
        result = loss(input, target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    """reduction='none'"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input = torch.tensor([[0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 1.0, 1.1, 1.2]])
        target = torch.tensor([0, 2, 3], dtype=torch.long)
        loss = torch.nn.modules.loss.MultiMarginLoss(reduction='none')
        result = loss(input, target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    """reduction='sum'"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input = torch.tensor([[0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 1.0, 1.1, 1.2]])
        target = torch.tensor([0, 2, 3], dtype=torch.long)
        loss = torch.nn.modules.loss.MultiMarginLoss(reduction='sum')
        result = loss(input, target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    """p=1, margin=0.5, weight, reduction='mean'"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input = torch.tensor([[0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 1.0, 1.1, 1.2]])
        target = torch.tensor([0, 2, 3], dtype=torch.long)
        weight = torch.tensor([0.5, 1.0, 1.5, 2.0])
        loss = torch.nn.modules.loss.MultiMarginLoss(p=1, margin=0.5, weight=weight, reduction='mean')
        result = loss(input, target)
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skip(
    reason="Positional args not aligning: Paddle lacks size_average/reduce params"
)
def test_case_8():
    """positional args"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input = torch.tensor([[0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 1.0, 1.1, 1.2]])
        target = torch.tensor([0, 2, 3], dtype=torch.long)
        loss = torch.nn.modules.loss.MultiMarginLoss(1, 0.5, None, 'none')
        result = loss(input, target)
        """
    )
    obj.run(pytorch_code, ["result"])
