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

obj = APIBase("torch.nn.functional.triplet_margin_with_distance_loss")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        import torch.nn as nn
        embedding = nn.Embedding(1000, 128)
        anchor_ids = torch.randint(0, 1000, (1,))
        anchor_ids = torch.tensor([568])
        positive_ids = torch.tensor([123])
        negative_ids = torch.tensor([121])
        anchor = embedding(anchor_ids)
        positive = embedding(positive_ids)
        negative = embedding(negative_ids)
        result = torch.nn.functional.triplet_margin_with_distance_loss(anchor, positive, negative, distance_function=nn.PairwiseDistance())
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        import torch.nn as nn
        embedding = nn.Embedding(1000, 128)
        anchor_ids = torch.randint(0, 1000, (1,))
        anchor_ids = torch.tensor([568])
        positive_ids = torch.tensor([123])
        negative_ids = torch.tensor([121])
        anchor = embedding(anchor_ids)
        positive = embedding(positive_ids)
        negative = embedding(negative_ids)
        result = torch.nn.functional.triplet_margin_with_distance_loss(anchor, positive, negative, distance_function=nn.PairwiseDistance(), reduction='mean')
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        import torch.nn as nn
        embedding = nn.Embedding(1000, 128)
        anchor_ids = torch.randint(0, 1000, (1,))
        anchor_ids = torch.tensor([568])
        positive_ids = torch.tensor([123])
        negative_ids = torch.tensor([121])
        anchor = embedding(anchor_ids)
        positive = embedding(positive_ids)
        negative = embedding(negative_ids)
        result = torch.nn.functional.triplet_margin_with_distance_loss(anchor, positive, negative, distance_function=nn.PairwiseDistance(), reduction='none')
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)
