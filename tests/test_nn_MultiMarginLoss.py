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

obj = APIBase("torch.nn.MultiMarginLoss")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        loss = torch.nn.MultiMarginLoss()
        input = torch.tensor([[1, -2, 3], [0, -1, 2], [1, 0, 1]]).to(dtype=torch.float32)
        label = torch.tensor([0, 1, 2])
        result = loss(input, label)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        loss = torch.nn.MultiMarginLoss(reduction='sum')
        input = torch.tensor([[1, -2, 3], [0, -1, 2], [1, 0, 1]]).to(dtype=torch.float32)
        label = torch.tensor([0, 1, 2])
        result = loss(input, label)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        loss = torch.nn.MultiMarginLoss(reduction='sum',
                p=1)
        input = torch.tensor([[1, -2, 3], [0, -1, 2], [1, 0, 1]]).to(dtype=torch.float32)
        label = torch.tensor([0, 1, 2])
        result = loss(input, label)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        loss = torch.nn.MultiMarginLoss(p=1, margin=1.0,
                weight=None, size_average=None,
                reduce=None, reduction='mean')
        input = torch.tensor([[1, -2, 3], [0, -1, 2], [1, 0, 1]]).to(dtype=torch.float32)
        label = torch.tensor([0, 1, 2])
        result = loss(input, label)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        loss = torch.nn.MultiMarginLoss(1, 1.0,
                None, None,
                None, 'mean')
        input = torch.tensor([[1, -2, 3], [0, -1, 2], [1, 0, 1]]).to(dtype=torch.float32)
        label = torch.tensor([0, 1, 2])
        result = loss(input, label)
        """
    )
    obj.run(pytorch_code, ["result"])
