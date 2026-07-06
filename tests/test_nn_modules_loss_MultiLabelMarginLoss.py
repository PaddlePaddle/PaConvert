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

from apibase import APIBase

obj = APIBase("torch.nn.modules.loss.MultiLabelMarginLoss")


def test_case_1():
    """default"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input = torch.tensor([[0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8]])
        target = torch.tensor([[0, 2, -1, -1],
            [1, 3, -1, -1]], dtype=torch.long)
        loss = torch.nn.modules.loss.MultiLabelMarginLoss()
        result = loss(input, target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    """reduction='none'"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input = torch.tensor([[0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8]])
        target = torch.tensor([[0, 2, -1, -1],
            [1, 3, -1, -1]], dtype=torch.long)
        loss = torch.nn.modules.loss.MultiLabelMarginLoss(reduction='none')
        result = loss(input, target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    """reduction='sum'"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input = torch.tensor([[0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8]])
        target = torch.tensor([[0, 2, -1, -1],
            [1, 3, -1, -1]], dtype=torch.long)
        loss = torch.nn.modules.loss.MultiLabelMarginLoss(reduction='sum')
        result = loss(input, target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    """reduction='mean'"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input = torch.tensor([[0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8]])
        target = torch.tensor([[0, 2, -1, -1],
            [1, 3, -1, -1]], dtype=torch.long)
        loss = torch.nn.modules.loss.MultiLabelMarginLoss(reduction='mean')
        result = loss(input, target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    """positional args"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input = torch.tensor([[0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8]])
        target = torch.tensor([[0, 2, -1, -1],
            [1, 3, -1, -1]], dtype=torch.long)
        loss = torch.nn.modules.loss.MultiLabelMarginLoss('mean')
        result = loss(input, target)
        """
    )
    obj.run(pytorch_code, ["result"])
