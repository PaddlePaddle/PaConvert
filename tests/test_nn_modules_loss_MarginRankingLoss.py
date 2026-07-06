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

obj = APIBase("torch.nn.modules.loss.MarginRankingLoss")


def test_case_1():
    """default"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input1 = torch.tensor([1.0, 2.0, 3.0])
        input2 = torch.tensor([4.0, 5.0, 6.0])
        target = torch.tensor([1.0, -1.0, 1.0])
        loss = torch.nn.modules.loss.MarginRankingLoss()
        result = loss(input1, input2, target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    """margin=0.5"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input1 = torch.tensor([1.0, 2.0, 3.0])
        input2 = torch.tensor([4.0, 5.0, 6.0])
        target = torch.tensor([1.0, -1.0, 1.0])
        loss = torch.nn.modules.loss.MarginRankingLoss(margin=0.5)
        result = loss(input1, input2, target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    """margin=1.0"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input1 = torch.tensor([1.0, 2.0, 3.0])
        input2 = torch.tensor([4.0, 5.0, 6.0])
        target = torch.tensor([1.0, -1.0, 1.0])
        loss = torch.nn.modules.loss.MarginRankingLoss(margin=1.0)
        result = loss(input1, input2, target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    """reduction='none'"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input1 = torch.tensor([1.0, 2.0, 3.0])
        input2 = torch.tensor([4.0, 5.0, 6.0])
        target = torch.tensor([1.0, -1.0, 1.0])
        loss = torch.nn.modules.loss.MarginRankingLoss(reduction='none')
        result = loss(input1, input2, target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    """reduction='sum'"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input1 = torch.tensor([1.0, 2.0, 3.0])
        input2 = torch.tensor([4.0, 5.0, 6.0])
        target = torch.tensor([1.0, -1.0, 1.0])
        loss = torch.nn.modules.loss.MarginRankingLoss(reduction='sum')
        result = loss(input1, input2, target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    """margin=0.5, reduction='mean'"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input1 = torch.tensor([1.0, 2.0, 3.0])
        input2 = torch.tensor([4.0, 5.0, 6.0])
        target = torch.tensor([1.0, -1.0, 1.0])
        loss = torch.nn.modules.loss.MarginRankingLoss(margin=0.5, reduction='mean')
        result = loss(input1, input2, target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    """positional args"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input1 = torch.tensor([1.0, 2.0, 3.0])
        input2 = torch.tensor([4.0, 5.0, 6.0])
        target = torch.tensor([1.0, -1.0, 1.0])
        loss = torch.nn.modules.loss.MarginRankingLoss(0.5, 'mean')
        result = loss(input1, input2, target)
        """
    )
    obj.run(pytorch_code, ["result"])
