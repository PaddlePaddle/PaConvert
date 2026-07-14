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

obj = APIBase("torch.Tensor.scatter_reduce_")


def test_case_1():
    """Basic usage with positional arguments"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        src = torch.tensor([1., 2., 3., 4., 5., 6.])
        index = torch.tensor([0, 1, 0, 1, 2, 1])
        input = torch.tensor([1., 2., 3., 4.])
        input.scatter_reduce_(0, index, src, reduce="sum")
        """
    )
    obj.run(pytorch_code, ["input"])


def test_case_2():
    """With include_self=False"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        src = torch.tensor([1., 2., 3., 4., 5., 6.])
        index = torch.tensor([0, 1, 0, 1, 2, 1])
        input = torch.tensor([1., 2., 3., 4.])
        input.scatter_reduce_(0, index, src, reduce="sum", include_self=False)
        """
    )
    obj.run(pytorch_code, ["input"])


def test_case_3():
    """With reduce='prod'"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[10., 20., 30.], [40., 50., 60.]])
        indices = torch.zeros((2, 3)).long()
        values = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
        input.scatter_reduce_(0, indices, values, "prod", include_self=True)
        """
    )
    obj.run(pytorch_code, ["input"])


def test_case_4():
    """With reduce='mean'"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[10., 20., 30.], [40., 50., 60.]])
        indices = torch.zeros((2, 3)).long()
        values = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
        input.scatter_reduce_(0, indices, values, "mean", include_self=True)
        """
    )
    obj.run(pytorch_code, ["input"])


def test_case_5():
    """With keyword arguments"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        src = torch.tensor([1., 2., 3., 4., 5., 6.])
        index = torch.tensor([0, 1, 0, 1, 2, 1])
        input = torch.tensor([1., 2., 3., 4.])
        input.scatter_reduce_(dim=0, index=index, src=src, reduce="sum")
        """
    )
    obj.run(pytorch_code, ["input"])


def test_case_6():
    """2D tensor with dim=1"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.zeros(2, 5)
        index = torch.tensor([[0, 1, 2, 0, 0], [0, 1, 2, 0, 0]])
        src = torch.arange(1, 11).reshape(2, 5).float()
        input.scatter_reduce_(1, index, src, reduce="sum", include_self=False)
        """
    )
    obj.run(pytorch_code, ["input"])
