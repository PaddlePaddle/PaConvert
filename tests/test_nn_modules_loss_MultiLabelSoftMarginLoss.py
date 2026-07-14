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

obj = APIBase("torch.nn.modules.loss.MultiLabelSoftMarginLoss")


def test_case_1():
    """default"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]])
        target = torch.tensor([[1, 0, 1],
            [0, 1, 0]], dtype=torch.float32)
        loss = torch.nn.modules.loss.MultiLabelSoftMarginLoss()
        result = loss(input, target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    """weight specified"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]])
        target = torch.tensor([[1, 0, 1],
            [0, 1, 0]], dtype=torch.float32)
        weight = torch.tensor([0.5, 1.0, 1.5])
        loss = torch.nn.modules.loss.MultiLabelSoftMarginLoss(weight=weight)
        result = loss(input, target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    """reduction='none'"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]])
        target = torch.tensor([[1, 0, 1],
            [0, 1, 0]], dtype=torch.float32)
        loss = torch.nn.modules.loss.MultiLabelSoftMarginLoss(reduction='none')
        result = loss(input, target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    """reduction='sum'"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]])
        target = torch.tensor([[1, 0, 1],
            [0, 1, 0]], dtype=torch.float32)
        loss = torch.nn.modules.loss.MultiLabelSoftMarginLoss(reduction='sum')
        result = loss(input, target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    """weight and reduction='mean'"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]])
        target = torch.tensor([[1, 0, 1],
            [0, 1, 0]], dtype=torch.float32)
        weight = torch.tensor([0.5, 1.0, 1.5])
        loss = torch.nn.modules.loss.MultiLabelSoftMarginLoss(weight=weight, reduction='mean')
        result = loss(input, target)
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skip(
    reason="Positional args not aligning: Paddle lacks size_average/reduce params"
)
def test_case_6():
    """positional args"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]])
        target = torch.tensor([[1, 0, 1],
            [0, 1, 0]], dtype=torch.float32)
        loss = torch.nn.modules.loss.MultiLabelSoftMarginLoss(None, 'sum')
        result = loss(input, target)
        """
    )
    obj.run(pytorch_code, ["result"])
