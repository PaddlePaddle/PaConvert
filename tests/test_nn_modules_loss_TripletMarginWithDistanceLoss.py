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

obj = APIBase("torch.nn.modules.loss.TripletMarginWithDistanceLoss")


def test_case_1():
    """default"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        anchor = torch.tensor([[1., 5, 3, 0], [0, 3, 2, 1]])
        positive = torch.tensor([[5., 1, 2, 0], [3, 2, 1, 0]])
        negative = torch.tensor([[2., 1, -3, 0], [1, 1, -1, 0]])
        model = nn.TripletMarginWithDistanceLoss()
        result = model(anchor, positive, negative)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    """margin=2"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        anchor = torch.tensor([[1., 5, 3, 0], [0, 3, 2, 1]])
        positive = torch.tensor([[5., 1, 2, 0], [3, 2, 1, 0]])
        negative = torch.tensor([[2., 1, -3, 0], [1, 1, -1, 0]])
        model = nn.TripletMarginWithDistanceLoss(margin=2)
        result = model(anchor, positive, negative)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    """margin=2, swap=True"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        anchor = torch.tensor([[1., 5, 3, 0], [0, 3, 2, 1]])
        positive = torch.tensor([[5., 1, 2, 0], [3, 2, 1, 0]])
        negative = torch.tensor([[2., 1, -3, 0], [1, 1, -1, 0]])
        model = nn.TripletMarginWithDistanceLoss(margin=2, swap=True)
        result = model(anchor, positive, negative)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    """margin=2, reduction='mean'"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        anchor = torch.tensor([[1., 5, 3, 0], [0, 3, 2, 1]])
        positive = torch.tensor([[5., 1, 2, 0], [3, 2, 1, 0]])
        negative = torch.tensor([[2., 1, -3, 0], [1, 1, -1, 0]])
        model = nn.TripletMarginWithDistanceLoss(margin=2, reduction='mean')
        result = model(anchor, positive, negative)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    """margin=2, reduction='sum'"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        anchor = torch.tensor([[1., 5, 3, 0], [0, 3, 2, 1]])
        positive = torch.tensor([[5., 1, 2, 0], [3, 2, 1, 0]])
        negative = torch.tensor([[2., 1, -3, 0], [1, 1, -1, 0]])
        model = nn.TripletMarginWithDistanceLoss(margin=2, reduction='sum')
        result = model(anchor, positive, negative)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    """margin=2, reduction='none'"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        anchor = torch.tensor([[1., 5, 3, 0], [0, 3, 2, 1]])
        positive = torch.tensor([[5., 1, 2, 0], [3, 2, 1, 0]])
        negative = torch.tensor([[2., 1, -3, 0], [1, 1, -1, 0]])
        model = nn.TripletMarginWithDistanceLoss(margin=2, reduction='none')
        result = model(anchor, positive, negative)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    """distance_function=nn.PairwiseDistance, margin=2, reduction='sum', swap=False"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        anchor = torch.tensor([[1., 5, 3, 0], [0, 3, 2, 1]])
        positive = torch.tensor([[5., 1, 2, 0], [3, 2, 1, 0]])
        negative = torch.tensor([[2., 1, -3, 0], [1, 1, -1, 0]])
        distance_function = nn.PairwiseDistance()
        model = nn.TripletMarginWithDistanceLoss(distance_function=distance_function, margin=2, reduction='sum', swap=False)
        result = model(anchor, positive, negative)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    """distance_function=nn.PairwiseDistance, margin=2, reduction='mean'"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        anchor = torch.tensor([[1., 5, 3, 0], [0, 3, 2, 1]])
        positive = torch.tensor([[5., 1, 2, 0], [3, 2, 1, 0]])
        negative = torch.tensor([[2., 1, -3, 0], [1, 1, -1, 0]])
        distance_function = nn.PairwiseDistance()
        model = nn.TripletMarginWithDistanceLoss(distance_function=distance_function, margin=2, reduction='mean')
        result = model(anchor, positive, negative)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    """all keyword args including distance_function=None"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        anchor = torch.tensor([[1., 5, 3, 0], [0, 3, 2, 1]])
        positive = torch.tensor([[5., 1, 2, 0], [3, 2, 1, 0]])
        negative = torch.tensor([[2., 1, -3, 0], [1, 1, -1, 0]])
        model = nn.TripletMarginWithDistanceLoss(distance_function=None, margin=2, reduction='sum', swap=False)
        result = model(anchor, positive, negative)
        """
    )
    obj.run(pytorch_code, ["result"])
