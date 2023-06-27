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

obj = APIBase("torch.nn.functional.multi_margin_loss")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[-1.2837, -0.0297,  0.0355],
            [ 0.9112, -1.7526, -0.4061]])
        target = torch.tensor([[1., 0., 1.],[0., 1., 1.]])
        result = torch.nn.functional.multi_margin_loss(input, target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[-1.2837, -0.0297,  0.0355],
            [ 0.9112, -1.7526, -0.4061]])
        target = torch.tensor([[1., 0., 1.],[0., 1., 1.]])
        result = torch.nn.functional.multi_margin_loss(input, target, reduction='sum')
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[-1.2837, -0.0297,  0.0355],
            [ 0.9112, -1.7526, -0.4061]])
        target = torch.tensor([[1., 0., 1.],[0., 1., 1.]])
        result = torch.nn.functional.multi_margin_loss(input, target, reduction='none')
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[-1.2837, -0.0297,  0.0355],
            [ 0.9112, -1.7526, -0.4061]])
        target = torch.tensor([[1., 0., 1.],[0., 1., 1.]])
        weight = torch.tensor([0.2, 0.3, 0.5])
        result = torch.nn.functional.multi_margin_loss(input, target, weight=weight)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[-1.2837, -0.0297,  0.0355],
            [ 0.9112, -1.7526, -0.4061]])
        target = torch.tensor([[1., 0., 1.],[0., 1., 1.]])
        result = torch.nn.functional.multi_margin_loss(input, target, size_average=False)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[-1.2837, -0.0297,  0.0355],
            [ 0.9112, -1.7526, -0.4061]])
        target = torch.tensor([[1., 0., 1.],[0., 1., 1.]])
        result = torch.nn.functional.multi_margin_loss(input, target, reduce=False)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[-1.2837, -0.0297,  0.0355],
            [ 0.9112, -1.7526, -0.4061]])
        target = torch.tensor([[1., 0., 1.],[0., 1., 1.]])
        result = torch.nn.functional.multi_margin_loss(input, target, margin=2)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[-1.2837, -0.0297,  0.0355],
            [ 0.9112, -1.7526, -0.4061]])
        target = torch.tensor([[1., 0., 1.],[0., 1., 1.]])
        result = torch.nn.functional.multi_margin_loss(input, target, p=2)
        """
    )
    obj.run(pytorch_code, ["result"])
