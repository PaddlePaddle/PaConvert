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

obj = APIBase("torch.nn.GaussianNLLLoss")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        loss = torch.nn.GaussianNLLLoss()
        input = torch.ones([5, 2]).to(dtype=torch.float32)
        label = torch.ones([5, 2]).to(dtype=torch.float32)
        variance = torch.ones([5, 2]).to(dtype=torch.float32)
        result = loss(input, label, variance)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        loss = torch.nn.GaussianNLLLoss(full=False)
        input = torch.ones([5, 2]).to(dtype=torch.float32)
        label = torch.ones([5, 2]).to(dtype=torch.float32)
        variance = torch.ones([5, 2]).to(dtype=torch.float32)
        result = loss(input, label, variance)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        loss = torch.nn.GaussianNLLLoss(eps=1e-08,
                full=False)
        input = torch.full([5, 2], 1).to(dtype=torch.float32)
        label = torch.full([5, 2], 2).to(dtype=torch.float32)
        variance = torch.ones([5, 2]).to(dtype=torch.float32)
        result = loss(input, label, variance)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        loss = torch.nn.GaussianNLLLoss(full=False,
                eps=1e-08,
                reduction='mean')
        input = torch.full([5, 2], 1).to(dtype=torch.float32)
        label = torch.full([5, 2], 2).to(dtype=torch.float32)
        variance = torch.ones([5, 2]).to(dtype=torch.float32)
        result = loss(input, label, variance)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        loss = torch.nn.GaussianNLLLoss(full=False,
                eps=1e-08,
                reduction='sum')
        input = torch.full([5, 2], 1).to(dtype=torch.float32)
        label = torch.full([5, 2], 2).to(dtype=torch.float32)
        variance = torch.ones([5, 2]).to(dtype=torch.float32)
        result = loss(input, label, variance)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    """All keyword arguments, reduction='none'"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        loss = torch.nn.GaussianNLLLoss(full=True, eps=1e-06, reduction='none')
        input = torch.tensor([[0.5, -0.3], [1.2, 0.7]], dtype=torch.float32)
        label = torch.tensor([[0.4, -0.2], [1.0, 0.5]], dtype=torch.float32)
        variance = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32)
        result = loss(input, label, variance)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    """Out-of-order keyword arguments"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        loss = torch.nn.GaussianNLLLoss(reduction='sum', eps=1e-08, full=False)
        input = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=torch.float32)
        label = torch.tensor([[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]], dtype=torch.float32)
        variance = torch.tensor([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]], dtype=torch.float32)
        result = loss(input, label, variance)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    """Default args (no kwargs)"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        loss = torch.nn.GaussianNLLLoss()
        input = torch.tensor([[1.5, 2.3], [-0.5, 1.7]], dtype=torch.float32)
        label = torch.tensor([[1.0, 2.0], [-0.3, 1.5]], dtype=torch.float32)
        variance = torch.tensor([[0.2, 0.3], [0.4, 0.5]], dtype=torch.float32)
        result = loss(input, label, variance)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    """3D inputs"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        loss = torch.nn.GaussianNLLLoss(full=False, eps=1e-06, reduction='mean')
        input = torch.ones([2, 3, 4], dtype=torch.float32) * 0.5
        label = torch.ones([2, 3, 4], dtype=torch.float32)
        variance = torch.ones([2, 3, 4], dtype=torch.float32) * 2.0
        result = loss(input, label, variance)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_10():
    """float64 dtype"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        loss = torch.nn.GaussianNLLLoss(eps=1e-10)
        input = torch.tensor([[0.5, -0.3], [1.2, 0.7]], dtype=torch.float64)
        label = torch.tensor([[0.4, -0.2], [1.0, 0.5]], dtype=torch.float64)
        variance = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float64)
        result = loss(input, label, variance)
        """
    )
    obj.run(pytorch_code, ["result"])
