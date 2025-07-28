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

obj = APIBase("torch.nn.MultiLabelMarginLoss")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        loss = torch.nn.MultiLabelMarginLoss(size_average=True, reduce=False, reduction="mean")
        input = torch.tensor([[1, -2, 3], [0, -1, 2], [1, 0, 1]]).to(dtype=torch.float32)
        label = torch.LongTensor([[-1, 1, -1], [1, 1, 1], [1, -1, 1]])
        result = loss(input, label)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        loss = torch.nn.MultiLabelMarginLoss(size_average=True, reduce=False, reduction="mean")
        input = torch.tensor([[0.1, 0.2, 0.4, 0.8], [0.2, 0.5, 0.3, 0.1]]).to(dtype=torch.float32)
        label = torch.LongTensor([[3, 0, -1, -1], [0, 2, -1, -1]])
        result = loss(input, label)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        loss = torch.nn.MultiLabelMarginLoss(size_average=True, reduce=False, reduction="sum")
        input = torch.tensor([[0.1, 0.2, 0.4, 0.8], [0.2, 0.5, 0.3, 0.1]]).to(dtype=torch.float32)
        label = torch.LongTensor([[3, 0, -1, -1], [0, 2, -1, -1]])
        result = loss(input, label)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        loss = torch.nn.MultiLabelMarginLoss(size_average=True, reduce=False, reduction="none")
        input = torch.tensor([[0.1, 0.2, 0.4, 0.8], [0.2, 0.5, 0.3, 0.1]]).to(dtype=torch.float32)
        label = torch.LongTensor([[3, 0, -1, -1], [0, 2, -1, -1]])
        result = loss(input, label)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        loss = torch.nn.MultiLabelMarginLoss(reduction="mean")
        input = torch.tensor([[0.1, 0.2, 0.4, 0.8]]).to(dtype=torch.float32)
        label = torch.LongTensor([[3, 0, -1, -1]])
        result = loss(input, label)
        """
    )
    obj.run(pytorch_code, ["result"])


def _test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        loss = torch.nn.MultiLabelMarginLoss(reduction="none")
        input = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]).to(dtype=torch.float32)
        label = torch.LongTensor([[-1, -1, -1], [-1, -1, -1]])
        result = loss(input, label)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        loss = torch.nn.MultiLabelMarginLoss(reduction="sum")
        input = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]).to(dtype=torch.float32)
        label = torch.LongTensor([[0, 1, 2], [0, 1, 2]])
        result = loss(input, label)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        loss = torch.nn.MultiLabelMarginLoss(size_average=None, reduce=True)
        input = torch.tensor([[0.1, 0.2, 0.4, 0.8], [0.2, 0.5, 0.3, 0.1]]).to(dtype=torch.float32)
        label = torch.LongTensor([[3, 0, -1, -1], [0, 2, -1, -1]])
        result = loss(input, label)
        """
    )
    obj.run(pytorch_code, ["result"])
