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

obj = APIBase("torch.nn.modules.loss.SoftMarginLoss")


def test_case_1():
    """default"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, -2, 3], [0, -1, 2]]).type(torch.float32)
        label = torch.Tensor([[-1, 1, -1], [1, 1, -1]]).type(torch.float32)
        cri = torch.nn.modules.loss.SoftMarginLoss()
        result = cri(input, label)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    """reduction='none'"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, -2, 3], [0, -1, 2]]).type(torch.float32)
        label = torch.Tensor([[-1, 1, -1], [1, 1, -1]]).type(torch.float32)
        cri = torch.nn.modules.loss.SoftMarginLoss(reduction='none')
        result = cri(input, label)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    """reduction='mean'"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, -2, 3], [0, -1, 2]]).type(torch.float32)
        label = torch.Tensor([[-1, 1, -1], [1, 1, -1]]).type(torch.float32)
        cri = torch.nn.modules.loss.SoftMarginLoss(reduction='mean')
        result = cri(input, label)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    """reduction='sum'"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, -2, 3], [0, -1, 2]]).type(torch.float32)
        label = torch.Tensor([[-1, 1, -1], [1, 1, -1]]).type(torch.float32)
        cri = torch.nn.modules.loss.SoftMarginLoss(reduction='sum')
        result = cri(input, label)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    """size_average=None, reduce=None, reduction='mean'"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, -2, 3], [0, -1, 2]]).type(torch.float32)
        label = torch.Tensor([[-1, 1, -1], [1, 1, -1]]).type(torch.float32)
        cri = torch.nn.modules.loss.SoftMarginLoss(size_average=None, reduce=None, reduction='mean')
        result = cri(input, label)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    """positional args: size_average, reduce, reduction"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, -2, 3], [0, -1, 2]]).type(torch.float32)
        label = torch.Tensor([[-1, 1, -1], [1, 1, -1]]).type(torch.float32)
        cri = torch.nn.modules.loss.SoftMarginLoss(None, None, 'mean')
        result = cri(input, label)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    """size_average=True, reduce=False, reduction='mean'"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, -2, 3], [0, -1, 2]]).type(torch.float32)
        label = torch.Tensor([[-1, 1, -1], [1, 1, -1]]).type(torch.float32)
        cri = torch.nn.modules.loss.SoftMarginLoss(size_average=True, reduce=False, reduction='mean')
        result = cri(input, label)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    """reduce=False, size_average=False, reduction='sum'"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, -2, 3], [0, -1, 2]]).type(torch.float32)
        label = torch.Tensor([[-1, 1, -1], [1, 1, -1]]).type(torch.float32)
        cri = torch.nn.modules.loss.SoftMarginLoss(reduce=False, size_average=False, reduction='sum')
        result = cri(input, label)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    """all keyword args: size_average=True, reduce=True, reduction='mean'"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, -2, 3], [0, -1, 2]]).type(torch.float32)
        label = torch.Tensor([[-1, 1, -1], [1, 1, -1]]).type(torch.float32)
        cri = torch.nn.modules.loss.SoftMarginLoss(size_average=True, reduce=True, reduction='mean')
        result = cri(input, label)
        """
    )
    obj.run(pytorch_code, ["result"])
