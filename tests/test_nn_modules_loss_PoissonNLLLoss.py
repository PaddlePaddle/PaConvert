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

obj = APIBase("torch.nn.modules.loss.PoissonNLLLoss")


def test_case_1():
    """default"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input = torch.tensor([[0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]])
        target = torch.tensor([[0.5, 1.0, 1.5],
            [2.0, 2.5, 3.0]])
        loss = torch.nn.modules.loss.PoissonNLLLoss()
        result = loss(input, target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    """log_input=False"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input = torch.tensor([[0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]])
        target = torch.tensor([[0.5, 1.0, 1.5],
            [2.0, 2.5, 3.0]])
        loss = torch.nn.modules.loss.PoissonNLLLoss(log_input=False)
        result = loss(input, target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    """full=True"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input = torch.tensor([[0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]])
        target = torch.tensor([[0.5, 1.0, 1.5],
            [2.0, 2.5, 3.0]])
        loss = torch.nn.modules.loss.PoissonNLLLoss(full=True)
        result = loss(input, target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    """eps=1e-08"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input = torch.tensor([[0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]])
        target = torch.tensor([[0.5, 1.0, 1.5],
            [2.0, 2.5, 3.0]])
        loss = torch.nn.modules.loss.PoissonNLLLoss(eps=1e-08)
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
        input = torch.tensor([[0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]])
        target = torch.tensor([[0.5, 1.0, 1.5],
            [2.0, 2.5, 3.0]])
        loss = torch.nn.modules.loss.PoissonNLLLoss(reduction='none')
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
        input = torch.tensor([[0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]])
        target = torch.tensor([[0.5, 1.0, 1.5],
            [2.0, 2.5, 3.0]])
        loss = torch.nn.modules.loss.PoissonNLLLoss(reduction='sum')
        result = loss(input, target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    """log_input=True, full=True, eps=1e-08, reduction='mean'"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input = torch.tensor([[0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]])
        target = torch.tensor([[0.5, 1.0, 1.5],
            [2.0, 2.5, 3.0]])
        loss = torch.nn.modules.loss.PoissonNLLLoss(log_input=True, full=True, eps=1e-08, reduction='mean')
        result = loss(input, target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    """log_input=False, reduction='mean'"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input = torch.tensor([[0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]])
        target = torch.tensor([[0.5, 1.0, 1.5],
            [2.0, 2.5, 3.0]])
        loss = torch.nn.modules.loss.PoissonNLLLoss(log_input=False, reduction='mean')
        result = loss(input, target)
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skip(
    reason="Positional args not aligning: Paddle uses epsilon not eps, lacks size_average/reduce"
)
def test_case_9():
    """positional args"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input = torch.tensor([[0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]])
        target = torch.tensor([[0.5, 1.0, 1.5],
            [2.0, 2.5, 3.0]])
        loss = torch.nn.modules.loss.PoissonNLLLoss(True, False, 1e-08, 'sum')
        result = loss(input, target)
        """
    )
    obj.run(pytorch_code, ["result"])
