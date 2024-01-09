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

obj = APIBase("torch.nn.MarginRankingLoss")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, 2], [3, 4]]).type(torch.float32)
        other = torch.Tensor([[2, 1], [2, 4]]).type(torch.float32)
        label = torch.Tensor([[1, -1], [-1, -1]]).type(torch.float32)
        margin_rank_loss = torch.nn.MarginRankingLoss()
        result = margin_rank_loss(input, other, label)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, 2], [3, 4]]).type(torch.float32)
        other = torch.Tensor([[2, 1], [2, 4]]).type(torch.float32)
        label = torch.Tensor([[1, -1], [-1, -1]]).type(torch.float32)
        margin_rank_loss = torch.nn.MarginRankingLoss(0.5, True, False, reduction='mean')
        result = margin_rank_loss(input, other, label)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, 2], [3, 4]]).type(torch.float32)
        other = torch.Tensor([[2, 1], [2, 4]]).type(torch.float32)
        label = torch.Tensor([[1, -1], [-1, -1]]).type(torch.float32)
        margin_rank_loss = torch.nn.MarginRankingLoss(0.7, reduce=False, size_average=False, reduction='mean')
        result = margin_rank_loss(input, other, label)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, 2], [3, 4]]).type(torch.float32)
        other = torch.Tensor([[2, 1], [2, 4]]).type(torch.float32)
        label = torch.Tensor([[1, -1], [-1, -1]]).type(torch.float32)
        margin_rank_loss = torch.nn.MarginRankingLoss(0.7, reduce=False, size_average=False, reduction='mean')
        result = margin_rank_loss(input, other, label)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, 2], [3, 4]]).type(torch.float32)
        other = torch.Tensor([[2, 1], [2, 4]]).type(torch.float32)
        label = torch.Tensor([[1, -1], [-1, -1]]).type(torch.float32)
        margin_rank_loss = torch.nn.MarginRankingLoss(0.7, reduce=False, size_average=False, reduction='mean')
        result = margin_rank_loss(input, other, label)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, 2], [3, 4]]).type(torch.float32)
        other = torch.Tensor([[2, 1], [2, 4]]).type(torch.float32)
        label = torch.Tensor([[1, -1], [-1, -1]]).type(torch.float32)
        margin_rank_loss = torch.nn.MarginRankingLoss(0.7, reduce=False, size_average=False, reduction='mean')
        result = margin_rank_loss(input, other, label)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, 2], [3, 4]]).type(torch.float32)
        other = torch.Tensor([[2, 1], [2, 4]]).type(torch.float32)
        label = torch.Tensor([[1, -1], [-1, -1]]).type(torch.float32)
        margin_rank_loss = torch.nn.MarginRankingLoss(0.7, reduce=False, size_average=True, reduction='mean')
        result = margin_rank_loss(input, other, label)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, 2], [3, 4]]).type(torch.float32)
        other = torch.Tensor([[2, 1], [2, 4]]).type(torch.float32)
        label = torch.Tensor([[1, -1], [-1, -1]]).type(torch.float32)
        margin_rank_loss = torch.nn.MarginRankingLoss(0.7, reduce=True, size_average=True, reduction='mean')
        result = margin_rank_loss(input, other, label)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, 2], [3, 4]]).type(torch.float32)
        other = torch.Tensor([[2, 1], [2, 4]]).type(torch.float32)
        label = torch.Tensor([[1, -1], [-1, -1]]).type(torch.float32)
        margin_rank_loss = torch.nn.MarginRankingLoss(0.7, reduce=False, size_average=False, reduction='sum')
        result = margin_rank_loss(input, other, label)
        """
    )
    obj.run(pytorch_code, ["result"])


# generated by validate_unittest autofix, based on test_case_2
def test_case_10():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, 2], [3, 4]]).type(torch.float32)
        other = torch.Tensor([[2, 1], [2, 4]]).type(torch.float32)
        label = torch.Tensor([[1, -1], [-1, -1]]).type(torch.float32)
        margin_rank_loss = torch.nn.MarginRankingLoss(0.5, True, False, 'mean')
        result = margin_rank_loss(input, other, label)
        """
    )
    obj.run(pytorch_code, ["result"])


# generated by validate_unittest autofix, based on test_case_2
def test_case_11():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, 2], [3, 4]]).type(torch.float32)
        other = torch.Tensor([[2, 1], [2, 4]]).type(torch.float32)
        label = torch.Tensor([[1, -1], [-1, -1]]).type(torch.float32)
        margin_rank_loss = torch.nn.MarginRankingLoss(margin=0.5, size_average=True, reduce=False, reduction='mean')
        result = margin_rank_loss(input, other, label)
        """
    )
    obj.run(pytorch_code, ["result"])


# generated by validate_unittest autofix, based on test_case_3
def test_case_12():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, 2], [3, 4]]).type(torch.float32)
        other = torch.Tensor([[2, 1], [2, 4]]).type(torch.float32)
        label = torch.Tensor([[1, -1], [-1, -1]]).type(torch.float32)
        margin_rank_loss = torch.nn.MarginRankingLoss(0.7, False, False, 'mean')
        result = margin_rank_loss(input, other, label)
        """
    )
    obj.run(pytorch_code, ["result"])


# generated by validate_unittest autofix, based on test_case_3
def test_case_13():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, 2], [3, 4]]).type(torch.float32)
        other = torch.Tensor([[2, 1], [2, 4]]).type(torch.float32)
        label = torch.Tensor([[1, -1], [-1, -1]]).type(torch.float32)
        margin_rank_loss = torch.nn.MarginRankingLoss(margin=0.7, size_average=False, reduce=False, reduction='mean')
        result = margin_rank_loss(input, other, label)
        """
    )
    obj.run(pytorch_code, ["result"])


# generated by validate_unittest autofix, based on test_case_4
def test_case_14():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, 2], [3, 4]]).type(torch.float32)
        other = torch.Tensor([[2, 1], [2, 4]]).type(torch.float32)
        label = torch.Tensor([[1, -1], [-1, -1]]).type(torch.float32)
        margin_rank_loss = torch.nn.MarginRankingLoss(0.7, False, False, 'mean')
        result = margin_rank_loss(input, other, label)
        """
    )
    obj.run(pytorch_code, ["result"])


# generated by validate_unittest autofix, based on test_case_4
def test_case_15():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, 2], [3, 4]]).type(torch.float32)
        other = torch.Tensor([[2, 1], [2, 4]]).type(torch.float32)
        label = torch.Tensor([[1, -1], [-1, -1]]).type(torch.float32)
        margin_rank_loss = torch.nn.MarginRankingLoss(margin=0.7, size_average=False, reduce=False, reduction='mean')
        result = margin_rank_loss(input, other, label)
        """
    )
    obj.run(pytorch_code, ["result"])


# generated by validate_unittest autofix, based on test_case_5
def test_case_16():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, 2], [3, 4]]).type(torch.float32)
        other = torch.Tensor([[2, 1], [2, 4]]).type(torch.float32)
        label = torch.Tensor([[1, -1], [-1, -1]]).type(torch.float32)
        margin_rank_loss = torch.nn.MarginRankingLoss(0.7, False, False, 'mean')
        result = margin_rank_loss(input, other, label)
        """
    )
    obj.run(pytorch_code, ["result"])


# generated by validate_unittest autofix, based on test_case_5
def test_case_17():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, 2], [3, 4]]).type(torch.float32)
        other = torch.Tensor([[2, 1], [2, 4]]).type(torch.float32)
        label = torch.Tensor([[1, -1], [-1, -1]]).type(torch.float32)
        margin_rank_loss = torch.nn.MarginRankingLoss(margin=0.7, size_average=False, reduce=False, reduction='mean')
        result = margin_rank_loss(input, other, label)
        """
    )
    obj.run(pytorch_code, ["result"])


# generated by validate_unittest autofix, based on test_case_6
def test_case_18():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, 2], [3, 4]]).type(torch.float32)
        other = torch.Tensor([[2, 1], [2, 4]]).type(torch.float32)
        label = torch.Tensor([[1, -1], [-1, -1]]).type(torch.float32)
        margin_rank_loss = torch.nn.MarginRankingLoss(0.7, False, False, 'mean')
        result = margin_rank_loss(input, other, label)
        """
    )
    obj.run(pytorch_code, ["result"])


# generated by validate_unittest autofix, based on test_case_6
def test_case_19():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, 2], [3, 4]]).type(torch.float32)
        other = torch.Tensor([[2, 1], [2, 4]]).type(torch.float32)
        label = torch.Tensor([[1, -1], [-1, -1]]).type(torch.float32)
        margin_rank_loss = torch.nn.MarginRankingLoss(margin=0.7, size_average=False, reduce=False, reduction='mean')
        result = margin_rank_loss(input, other, label)
        """
    )
    obj.run(pytorch_code, ["result"])


# generated by validate_unittest autofix, based on test_case_7
def test_case_20():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, 2], [3, 4]]).type(torch.float32)
        other = torch.Tensor([[2, 1], [2, 4]]).type(torch.float32)
        label = torch.Tensor([[1, -1], [-1, -1]]).type(torch.float32)
        margin_rank_loss = torch.nn.MarginRankingLoss(0.7, True, False, 'mean')
        result = margin_rank_loss(input, other, label)
        """
    )
    obj.run(pytorch_code, ["result"])


# generated by validate_unittest autofix, based on test_case_7
def test_case_21():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, 2], [3, 4]]).type(torch.float32)
        other = torch.Tensor([[2, 1], [2, 4]]).type(torch.float32)
        label = torch.Tensor([[1, -1], [-1, -1]]).type(torch.float32)
        margin_rank_loss = torch.nn.MarginRankingLoss(margin=0.7, size_average=True, reduce=False, reduction='mean')
        result = margin_rank_loss(input, other, label)
        """
    )
    obj.run(pytorch_code, ["result"])


# generated by validate_unittest autofix, based on test_case_8
def test_case_22():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, 2], [3, 4]]).type(torch.float32)
        other = torch.Tensor([[2, 1], [2, 4]]).type(torch.float32)
        label = torch.Tensor([[1, -1], [-1, -1]]).type(torch.float32)
        margin_rank_loss = torch.nn.MarginRankingLoss(0.7, True, True, 'mean')
        result = margin_rank_loss(input, other, label)
        """
    )
    obj.run(pytorch_code, ["result"])


# generated by validate_unittest autofix, based on test_case_8
def test_case_23():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, 2], [3, 4]]).type(torch.float32)
        other = torch.Tensor([[2, 1], [2, 4]]).type(torch.float32)
        label = torch.Tensor([[1, -1], [-1, -1]]).type(torch.float32)
        margin_rank_loss = torch.nn.MarginRankingLoss(margin=0.7, size_average=True, reduce=True, reduction='mean')
        result = margin_rank_loss(input, other, label)
        """
    )
    obj.run(pytorch_code, ["result"])


# generated by validate_unittest autofix, based on test_case_9
def test_case_24():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, 2], [3, 4]]).type(torch.float32)
        other = torch.Tensor([[2, 1], [2, 4]]).type(torch.float32)
        label = torch.Tensor([[1, -1], [-1, -1]]).type(torch.float32)
        margin_rank_loss = torch.nn.MarginRankingLoss(0.7, False, False, 'sum')
        result = margin_rank_loss(input, other, label)
        """
    )
    obj.run(pytorch_code, ["result"])


# generated by validate_unittest autofix, based on test_case_9
def test_case_25():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, 2], [3, 4]]).type(torch.float32)
        other = torch.Tensor([[2, 1], [2, 4]]).type(torch.float32)
        label = torch.Tensor([[1, -1], [-1, -1]]).type(torch.float32)
        margin_rank_loss = torch.nn.MarginRankingLoss(margin=0.7, size_average=False, reduce=False, reduction='sum')
        result = margin_rank_loss(input, other, label)
        """
    )
    obj.run(pytorch_code, ["result"])
