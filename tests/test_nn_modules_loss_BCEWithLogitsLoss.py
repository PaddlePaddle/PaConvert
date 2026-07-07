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

obj = APIBase("torch.nn.modules.loss.BCEWithLogitsLoss")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[0.2837, 0.0297, 0.0355],
            [0.9112, 0.7526, 0.4061]])
        target = torch.tensor([[1.,0.,1.],[0.,1.,0.]])
        loss = torch.nn.modules.loss.BCEWithLogitsLoss()
        result = loss(input, target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[0.2837, 0.0297, 0.0355],
            [0.9112, 0.7526, 0.4061]])
        target = torch.tensor([[1.,0.,1.],[0.,1.,0.]])
        pos_weight = torch.tensor([0.5, 0.2, 0.3])
        loss = torch.nn.modules.loss.BCEWithLogitsLoss(pos_weight=pos_weight)
        result = loss(input, target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[0.2837, 0.0297, 0.0355],
            [0.9112, 0.7526, 0.4061]])
        target = torch.tensor([[1.,0.,1.],[0.,1.,0.]])
        loss = torch.nn.modules.loss.BCEWithLogitsLoss(reduction='none')
        result = loss(input, target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[0.2837, 0.0297, 0.0355],
            [0.9112, 0.7526, 0.4061]])
        target = torch.tensor([[1.,0.,1.],[0.,1.,0.]])
        loss = torch.nn.modules.loss.BCEWithLogitsLoss(reduction='mean')
        result = loss(input, target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[0.2837, 0.0297, 0.0355],
            [0.9112, 0.7526, 0.4061]])
        target = torch.tensor([[1.,0.,1.],[0.,1.,0.]])
        loss = torch.nn.modules.loss.BCEWithLogitsLoss(reduction='sum')
        result = loss(input, target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[0.2837, 0.0297, 0.0355],
            [0.9112, 0.7526, 0.4061]])
        target = torch.tensor([[1.,0.,1.],[0.,1.,0.]])
        pos_weight = torch.tensor([0.5, 0.2, 0.3])
        loss = torch.nn.modules.loss.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
        result = loss(input, target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[0.2837, 0.0297, 0.0355],
            [0.9112, 0.7526, 0.4061]])
        target = torch.tensor([[1.,0.,1.],[0.,1.,0.]])
        pos_weight = torch.tensor([0.5, 0.2, 0.3])
        loss = torch.nn.modules.loss.BCEWithLogitsLoss(pos_weight=pos_weight, size_average=True)
        result = loss(input, target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[0.2837, 0.0297, 0.0355],
            [0.9112, 0.7526, 0.4061]])
        target = torch.tensor([[1.,0.,1.],[0.,1.,0.]])
        pos_weight = torch.tensor([0.5, 0.2, 0.3])
        loss = torch.nn.modules.loss.BCEWithLogitsLoss(pos_weight=pos_weight, size_average=False)
        result = loss(input, target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[0.2837, 0.0297, 0.0355],
            [0.9112, 0.7526, 0.4061]])
        target = torch.tensor([[1.,0.,1.],[0.,1.,0.]])
        pos_weight = torch.tensor([0.5, 0.2, 0.3])
        loss = torch.nn.modules.loss.BCEWithLogitsLoss(pos_weight=pos_weight, reduce=True)
        result = loss(input, target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_10():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[0.2837, 0.0297, 0.0355],
            [0.9112, 0.7526, 0.4061]])
        target = torch.tensor([[1.,0.,1.],[0.,1.,0.]])
        pos_weight = torch.tensor([0.5, 0.2, 0.3])
        loss = torch.nn.modules.loss.BCEWithLogitsLoss(pos_weight=pos_weight, reduce=False)
        result = loss(input, target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_11():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[0.2837, 0.0297, 0.0355],
            [0.9112, 0.7526, 0.4061]])
        target = torch.tensor([[1.,0.,1.],[0.,1.,0.]])
        pos_weight = torch.tensor([0.5, 0.2, 0.3])
        loss = torch.nn.modules.loss.BCEWithLogitsLoss(pos_weight=pos_weight, size_average=None, reduce=False, reduction='mean')
        result = loss(input, target)
        """
    )
    obj.run(pytorch_code, ["result"])


# generated by validate_unittest autofix, based on test_case_11
def test_case_12():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[0.2837, 0.0297, 0.0355],
            [0.9112, 0.7526, 0.4061]])
        target = torch.tensor([[1.,0.,1.],[0.,1.,0.]])
        pos_weight = torch.tensor([0.5, 0.2, 0.3])
        loss = torch.nn.modules.loss.BCEWithLogitsLoss(pos_weight, None, False, 'mean')
        result = loss(input, target)
        """
    )
    obj.run(pytorch_code, ["result"])


# generated by validate_unittest autofix, based on test_case_11
def test_case_13():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[0.2837, 0.0297, 0.0355],
            [0.9112, 0.7526, 0.4061]])
        target = torch.tensor([[1.,0.,1.],[0.,1.,0.]])
        pos_weight = torch.tensor([0.5, 0.2, 0.3])
        loss = torch.nn.modules.loss.BCEWithLogitsLoss(reduction='mean', reduce=False, size_average=None, pos_weight=pos_weight)
        result = loss(input, target)
        """
    )
    obj.run(pytorch_code, ["result"])
