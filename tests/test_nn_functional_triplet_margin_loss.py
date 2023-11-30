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

obj = APIBase("torch.nn.functional.triplet_margin_loss")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        input = torch.tensor([[1, 5, 3], [0, 3, 2], [1, 4, 1]]).to(dtype=torch.float32)
        positive = torch.tensor([[5, 1, 2], [3, 2, 1], [3, -1, 1]]).to(dtype=torch.float32)
        negative = torch.tensor([[2, 1, -3], [1, 1, -1], [4, -2, 1]]).to(dtype=torch.float32)
        result = F.triplet_margin_loss(input, positive, negative, margin=1.0, reduction='none')
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        input = torch.tensor([[1, 5, 3], [0, 3, 2], [1, 4, 1]]).to(dtype=torch.float32)
        positive = torch.tensor([[5, 1, 2], [3, 2, 1], [3, -1, 1]]).to(dtype=torch.float32)
        negative = torch.tensor([[2, 1, -3], [1, 1, -1], [4, -2, 1]]).to(dtype=torch.float32)
        result = F.triplet_margin_loss(input, positive, negative, 1.0, 2)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        input = torch.tensor([[1, 5, 3], [0, 3, 2], [1, 4, 1]]).to(dtype=torch.float32)
        positive = torch.tensor([[5, 1, 2], [3, 2, 1], [3, -1, 1]]).to(dtype=torch.float32)
        negative = torch.tensor([[2, 1, -3], [1, 1, -1], [4, -2, 1]]).to(dtype=torch.float32)
        result = F.triplet_margin_loss(margin=1.0, reduction='none',
                anchor=input, positive=positive, negative=negative)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        input = torch.tensor([[1, 5, 3], [0, 3, 2], [1, 4, 1]]).to(dtype=torch.float32)
        positive = torch.tensor([[5, 1, 2], [3, 2, 1], [3, -1, 1]]).to(dtype=torch.float32)
        negative = torch.tensor([[2, 1, -3], [1, 1, -1], [4, -2, 1]]).to(dtype=torch.float32)
        result = F.triplet_margin_loss(anchor=input,
                positive = positive,
                negative = negative,
                margin=1.0, p=2, eps=1e-06,
                swap=False, size_average=None,
                reduce=None, reduction='mean')
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        input = torch.tensor([[1, 5, 3], [0, 3, 2], [1, 4, 1]]).to(dtype=torch.float32)
        positive = torch.tensor([[5, 1, 2], [3, 2, 1], [3, -1, 1]]).to(dtype=torch.float32)
        negative = torch.tensor([[2, 1, -3], [1, 1, -1], [4, -2, 1]]).to(dtype=torch.float32)
        result = F.triplet_margin_loss(input,
                positive,
                negative,
                1.0, 2, 1e-06,
                False, None,
                None, 'mean')
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        input = torch.tensor([[1, 5, 3], [0, 3, 2], [1, 4, 1]]).to(dtype=torch.float32)
        positive = torch.tensor([[5, 1, 2], [3, 2, 1], [3, -1, 1]]).to(dtype=torch.float32)
        negative = torch.tensor([[2, 1, -3], [1, 1, -1], [4, -2, 1]]).to(dtype=torch.float32)
        result = F.triplet_margin_loss(input, positive, negative)
        """
    )
    obj.run(pytorch_code, ["result"])
