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

obj = APIBase("torch.nn.modules.loss.MSELoss")


def test_case_1():
    """default"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[-1.2837, -0.0297,  0.0355],
            [ 0.9112, -1.7526, -0.4061]])
        target = torch.tensor([[1.,2.,3.],[4.,5.,6.]])
        loss = torch.nn.modules.loss.MSELoss()
        result = loss(input,target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    """reduction='none'"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[-1.2837, -0.0297,  0.0355],
            [ 0.9112, -1.7526, -0.4061]])
        target = torch.tensor([[1.,2.,3.],[4.,5.,6.]])
        loss = torch.nn.modules.loss.MSELoss(reduction='none')
        result = loss(input,target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    """reduction='mean'"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[-1.2837, -0.0297,  0.0355],
            [ 0.9112, -1.7526, -0.4061]])
        target = torch.tensor([[1.,2.,3.],[4.,5.,6.]])
        loss = torch.nn.modules.loss.MSELoss(reduction='mean')
        result = loss(input,target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    """reduction='sum'"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[-1.2837, -0.0297,  0.0355],
            [ 0.9112, -1.7526, -0.4061]])
        target = torch.tensor([[1.,2.,3.],[4.,5.,6.]])
        loss = torch.nn.modules.loss.MSELoss(reduction='sum')
        result = loss(input,target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    """size_average=True"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[-1.2837, -0.0297,  0.0355],
            [ 0.9112, -1.7526, -0.4061]])
        target = torch.tensor([[1.,2.,3.],[4.,5.,6.]])
        loss = torch.nn.modules.loss.MSELoss(size_average=True)
        result = loss(input,target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    """size_average=False"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[-1.2837, -0.0297,  0.0355],
            [ 0.9112, -1.7526, -0.4061]])
        target = torch.tensor([[1.,2.,3.],[4.,5.,6.]])
        loss = torch.nn.modules.loss.MSELoss(size_average=False)
        result = loss(input,target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    """reduce=True"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[-1.2837, -0.0297,  0.0355],
            [ 0.9112, -1.7526, -0.4061]])
        target = torch.tensor([[1.,2.,3.],[4.,5.,6.]])
        loss = torch.nn.modules.loss.MSELoss(reduce=True)
        result = loss(input,target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    """reduce=False"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[-1.2837, -0.0297,  0.0355],
            [ 0.9112, -1.7526, -0.4061]])
        target = torch.tensor([[1.,2.,3.],[4.,5.,6.]])
        loss = torch.nn.modules.loss.MSELoss(reduce=False)
        result = loss(input,target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    """positional args"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[-1.2837, -0.0297,  0.0355],
            [ 0.9112, -1.7526, -0.4061]])
        target = torch.tensor([[1.,2.,3.],[4.,5.,6.]])
        loss = torch.nn.modules.loss.MSELoss(None, False, 'mean')
        result = loss(input,target)
        """
    )
    obj.run(pytorch_code, ["result"])
