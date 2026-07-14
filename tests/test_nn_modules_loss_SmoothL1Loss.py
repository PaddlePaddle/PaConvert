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

obj = APIBase("torch.nn.modules.loss.SmoothL1Loss")


def test_case_1():
    """default"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        loss = torch.nn.modules.loss.SmoothL1Loss()
        input = torch.ones([2, 3]).to(dtype=torch.float32)
        label = torch.full([2, 3], 2).to(dtype=torch.float32)
        result = loss(input, label)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    """reduction='none'"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        loss = torch.nn.modules.loss.SmoothL1Loss(reduction='none')
        input = torch.ones([2, 3]).to(dtype=torch.float32)
        label = torch.full([2, 3], 2).to(dtype=torch.float32)
        result = loss(input, label)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    """reduction='mean'"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        loss = torch.nn.modules.loss.SmoothL1Loss(reduction='mean')
        input = torch.ones([2, 3]).to(dtype=torch.float32)
        label = torch.full([2, 3], 2).to(dtype=torch.float32)
        result = loss(input, label)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    """reduction='sum'"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        loss = torch.nn.modules.loss.SmoothL1Loss(reduction='sum')
        input = torch.ones([2, 3]).to(dtype=torch.float32)
        label = torch.full([2, 3], 2).to(dtype=torch.float32)
        result = loss(input, label)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    """beta=1.0, reduction='none'"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        loss = torch.nn.modules.loss.SmoothL1Loss(beta=1.0, reduction='none')
        input = torch.ones([2, 3]).to(dtype=torch.float32)
        label = torch.full([2, 3], 2).to(dtype=torch.float32)
        result = loss(input, label)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    """size_average=None, reduce=None, reduction='mean', beta=1.0"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        loss = torch.nn.modules.loss.SmoothL1Loss(size_average=None,
                reduce=None, reduction='mean', beta=1.0)
        input = torch.ones([2, 3]).to(dtype=torch.float32)
        label = torch.full([2, 3], 2).to(dtype=torch.float32)
        result = loss(input, label)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    """positional args"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        loss = torch.nn.modules.loss.SmoothL1Loss(None,
                None, 'mean', 1.0)
        input = torch.ones([2, 3]).to(dtype=torch.float32)
        label = torch.full([2, 3], 2).to(dtype=torch.float32)
        result = loss(input, label)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    """beta=1.5, reduction='none'"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        loss = torch.nn.modules.loss.SmoothL1Loss(beta=1.5, reduction='none')
        input = torch.ones([2, 3]).to(dtype=torch.float32)
        label = torch.full([2, 3], 2).to(dtype=torch.float32)
        result = loss(input, label)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    """beta as variable, reduction='none'"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        beta = 1.5
        loss = torch.nn.modules.loss.SmoothL1Loss(beta=beta, reduction='none')
        input = torch.ones([2, 3]).to(dtype=torch.float32)
        label = torch.full([2, 3], 2).to(dtype=torch.float32)
        result = loss(input, label)
        """
    )
    obj.run(pytorch_code, ["result"])
