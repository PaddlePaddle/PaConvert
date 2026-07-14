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

obj = APIBase("torch.nn.modules.loss.TripletMarginLoss")


def test_case_1():
    """default"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        anchor = torch.Tensor([[1, 5, 3, 0], [0, 3, 2, 1]]).type(torch.float32)
        positive = torch.Tensor([[5, 1, 2, 0], [3, 2, 1, 0]]).type(torch.float32)
        negative = torch.Tensor([[2, 1, -3, 0], [1, 1, -1, 0]]).type(torch.float32)
        cri = torch.nn.modules.loss.TripletMarginLoss()
        result = cri(anchor, positive, negative)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    """margin=1.3, reduction='none'"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        anchor = torch.Tensor([[1, 5, 3, 0], [0, 3, 2, 1]]).type(torch.float32)
        positive = torch.Tensor([[5, 1, 2, 0], [3, 2, 1, 0]]).type(torch.float32)
        negative = torch.Tensor([[2, 1, -3, 0], [1, 1, -1, 0]]).type(torch.float32)
        cri = torch.nn.modules.loss.TripletMarginLoss(margin=1.3, reduction='none')
        result = cri(anchor, positive, negative)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    """p=2, reduction='mean'"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        anchor = torch.Tensor([[1, 5, 3, 0], [0, 3, 2, 1]]).type(torch.float32)
        positive = torch.Tensor([[5, 1, 2, 0], [3, 2, 1, 0]]).type(torch.float32)
        negative = torch.Tensor([[2, 1, -3, 0], [1, 1, -1, 0]]).type(torch.float32)
        cri = torch.nn.modules.loss.TripletMarginLoss(p=2, reduction='mean')
        result = cri(anchor, positive, negative)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    """eps=1e-4, reduction='sum'"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        anchor = torch.Tensor([[1, 5, 3, 0], [0, 3, 2, 1]]).type(torch.float32)
        positive = torch.Tensor([[5, 1, 2, 0], [3, 2, 1, 0]]).type(torch.float32)
        negative = torch.Tensor([[2, 1, -3, 0], [1, 1, -1, 0]]).type(torch.float32)
        cri = torch.nn.modules.loss.TripletMarginLoss(eps=1e-4, reduction='sum')
        result = cri(anchor, positive, negative)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    """swap=True, reduction='none'"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        anchor = torch.Tensor([[1, 5, 3, 0], [0, 3, 2, 1]]).type(torch.float32)
        positive = torch.Tensor([[5, 1, 2, 0], [3, 2, 1, 0]]).type(torch.float32)
        negative = torch.Tensor([[2, 1, -3, 0], [1, 1, -1, 0]]).type(torch.float32)
        cri = torch.nn.modules.loss.TripletMarginLoss(swap=True, reduction='none')
        result = cri(anchor, positive, negative)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    """all keyword args"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        anchor = torch.Tensor([[1, 5, 3, 0], [0, 3, 2, 1]]).type(torch.float32)
        positive = torch.Tensor([[5, 1, 2, 0], [3, 2, 1, 0]]).type(torch.float32)
        negative = torch.Tensor([[2, 1, -3, 0], [1, 1, -1, 0]]).type(torch.float32)
        cri = torch.nn.modules.loss.TripletMarginLoss(margin=1.3, p=3.2, eps=1e-5, swap=False, size_average=True, reduce=False, reduction='mean')
        result = cri(anchor, positive, negative)
        """
    )
    obj.run(pytorch_code, ["result"], rtol=1.0e-5, atol=1.0e-8)


def test_case_7():
    """positional args"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        anchor = torch.Tensor([[1, 5, 3, 0], [0, 3, 2, 1]]).type(torch.float32)
        positive = torch.Tensor([[5, 1, 2, 0], [3, 2, 1, 0]]).type(torch.float32)
        negative = torch.Tensor([[2, 1, -3, 0], [1, 1, -1, 0]]).type(torch.float32)
        cri = torch.nn.modules.loss.TripletMarginLoss(1.3, 3.2, 1e-5, False, True, False, reduction='mean')
        result = cri(anchor, positive, negative)
        """
    )
    obj.run(pytorch_code, ["result"], rtol=1.0e-5, atol=1.0e-8)


def test_case_8():
    """all positional args"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        anchor = torch.Tensor([[1, 5, 3, 0], [0, 3, 2, 1]]).type(torch.float32)
        positive = torch.Tensor([[5, 1, 2, 0], [3, 2, 1, 0]]).type(torch.float32)
        negative = torch.Tensor([[2, 1, -3, 0], [1, 1, -1, 0]]).type(torch.float32)
        cri = torch.nn.modules.loss.TripletMarginLoss(1.3, 3.2, 1e-5, True, True, False, 'none')
        result = cri(anchor, positive, negative)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    """reduction='sum', swap=True, size_average=False, reduce=False"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        anchor = torch.Tensor([[1, 5, 3, 0], [0, 3, 2, 1]]).type(torch.float32)
        positive = torch.Tensor([[5, 1, 2, 0], [3, 2, 1, 0]]).type(torch.float32)
        negative = torch.Tensor([[2, 1, -3, 0], [1, 1, -1, 0]]).type(torch.float32)
        cri = torch.nn.modules.loss.TripletMarginLoss(1.3, 3.2, 1e-5, True, False, True, reduction='sum')
        result = cri(anchor, positive, negative)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_10():
    """rearranged keyword args"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        anchor = torch.Tensor([[1, 5, 3, 0], [0, 3, 2, 1]]).type(torch.float32)
        positive = torch.Tensor([[5, 1, 2, 0], [3, 2, 1, 0]]).type(torch.float32)
        negative = torch.Tensor([[2, 1, -3, 0], [1, 1, -1, 0]]).type(torch.float32)
        cri = torch.nn.modules.loss.TripletMarginLoss(reduction='sum', reduce=False, size_average=False, swap=False, eps=1e-5, p=3.2, margin=1.3)
        result = cri(anchor, positive, negative)
        """
    )
    obj.run(pytorch_code, ["result"], rtol=1.0e-5, atol=1.0e-8)
