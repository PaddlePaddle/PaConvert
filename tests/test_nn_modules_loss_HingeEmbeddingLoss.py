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

obj = APIBase("torch.nn.modules.loss.HingeEmbeddingLoss")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, -2, 3, 0, -1], [0, -1, 2, 1, 1], [1, 0, 1, -1, 2]]).type(torch.float32)
        label = torch.Tensor([[-1, 1, -1, 1, 1], [1, 1, 1, -1, -1], [1, -1, 1, 1, -1]]).type(torch.float32)
        loss = torch.nn.modules.loss.HingeEmbeddingLoss()
        result = loss(input, label)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, -2, 3, 0, -1], [0, -1, 2, 1, 1], [1, 0, 1, -1, 2]]).type(torch.float32)
        label = torch.Tensor([[-1, 1, -1, 1, 1], [1, 1, 1, -1, -1], [1, -1, 1, 1, -1]]).type(torch.float32)
        loss = torch.nn.modules.loss.HingeEmbeddingLoss(margin=0.5)
        result = loss(input, label)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, -2, 3, 0, -1], [0, -1, 2, 1, 1], [1, 0, 1, -1, 2]]).type(torch.float32)
        label = torch.Tensor([[-1, 1, -1, 1, 1], [1, 1, 1, -1, -1], [1, -1, 1, 1, -1]]).type(torch.float32)
        loss = torch.nn.modules.loss.HingeEmbeddingLoss(margin=0.5, reduction='none')
        result = loss(input, label)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, -2, 3, 0, -1], [0, -1, 2, 1, 1], [1, 0, 1, -1, 2]]).type(torch.float32)
        label = torch.Tensor([[-1, 1, -1, 1, 1], [1, 1, 1, -1, -1], [1, -1, 1, 1, -1]]).type(torch.float32)
        loss = torch.nn.modules.loss.HingeEmbeddingLoss(margin=0.5, reduction='mean')
        result = loss(input, label)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, -2, 3, 0, -1], [0, -1, 2, 1, 1], [1, 0, 1, -1, 2]]).type(torch.float32)
        label = torch.Tensor([[-1, 1, -1, 1, 1], [1, 1, 1, -1, -1], [1, -1, 1, 1, -1]]).type(torch.float32)
        loss = torch.nn.modules.loss.HingeEmbeddingLoss(margin=0.5, reduction='sum')
        result = loss(input, label)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, -2, 3, 0, -1], [0, -1, 2, 1, 1], [1, 0, 1, -1, 2]]).type(torch.float32)
        label = torch.Tensor([[-1, 1, -1, 1, 1], [1, 1, 1, -1, -1], [1, -1, 1, 1, -1]]).type(torch.float32)
        loss = torch.nn.modules.loss.HingeEmbeddingLoss(margin=0.5, size_average=True)
        result = loss(input, label)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, -2, 3, 0, -1], [0, -1, 2, 1, 1], [1, 0, 1, -1, 2]]).type(torch.float32)
        label = torch.Tensor([[-1, 1, -1, 1, 1], [1, 1, 1, -1, -1], [1, -1, 1, 1, -1]]).type(torch.float32)
        loss = torch.nn.modules.loss.HingeEmbeddingLoss(margin=0.5, size_average=False)
        result = loss(input, label)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, -2, 3, 0, -1], [0, -1, 2, 1, 1], [1, 0, 1, -1, 2]]).type(torch.float32)
        label = torch.Tensor([[-1, 1, -1, 1, 1], [1, 1, 1, -1, -1], [1, -1, 1, 1, -1]]).type(torch.float32)
        loss = torch.nn.modules.loss.HingeEmbeddingLoss(margin=0.5, reduce=True)
        result = loss(input, label)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, -2, 3, 0, -1], [0, -1, 2, 1, 1], [1, 0, 1, -1, 2]]).type(torch.float32)
        label = torch.Tensor([[-1, 1, -1, 1, 1], [1, 1, 1, -1, -1], [1, -1, 1, 1, -1]]).type(torch.float32)
        loss = torch.nn.modules.loss.HingeEmbeddingLoss(margin=0.5, reduce=False)
        result = loss(input, label)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_10():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, -2, 3, 0, -1], [0, -1, 2, 1, 1], [1, 0, 1, -1, 2]]).type(torch.float32)
        label = torch.Tensor([[-1, 1, -1, 1, 1], [1, 1, 1, -1, -1], [1, -1, 1, 1, -1]]).type(torch.float32)
        loss = torch.nn.modules.loss.HingeEmbeddingLoss(margin=0.5, size_average=None, reduce=False, reduction='mean')
        result = loss(input, label)
        """
    )
    obj.run(pytorch_code, ["result"])


# generated by validate_unittest autofix, based on test_case_10
def test_case_11():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, -2, 3, 0, -1], [0, -1, 2, 1, 1], [1, 0, 1, -1, 2]]).type(torch.float32)
        label = torch.Tensor([[-1, 1, -1, 1, 1], [1, 1, 1, -1, -1], [1, -1, 1, 1, -1]]).type(torch.float32)
        loss = torch.nn.modules.loss.HingeEmbeddingLoss(0.5, None, False, 'mean')
        result = loss(input, label)
        """
    )
    obj.run(pytorch_code, ["result"])


# generated by validate_unittest autofix, based on test_case_10
def test_case_12():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, -2, 3, 0, -1], [0, -1, 2, 1, 1], [1, 0, 1, -1, 2]]).type(torch.float32)
        label = torch.Tensor([[-1, 1, -1, 1, 1], [1, 1, 1, -1, -1], [1, -1, 1, 1, -1]]).type(torch.float32)
        loss = torch.nn.modules.loss.HingeEmbeddingLoss(reduction='mean', reduce=False, size_average=None, margin=0.5)
        result = loss(input, label)
        """
    )
    obj.run(pytorch_code, ["result"])
