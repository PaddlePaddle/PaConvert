# Copyright (c) 2023 torchtorch Authors. All Rights Reserved.
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

obj = APIBase("torch.nn.functional.gaussian_nll_loss")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.ones([5, 2], dtype=torch.float32)
        label = torch.ones([5, 2], dtype=torch.float32)
        variance = torch.ones([5, 2], dtype=torch.float32)
        result = torch.nn.functional.gaussian_nll_loss(input, label, variance)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.ones([5, 2], dtype=torch.float32)
        label = torch.ones([5, 2], dtype=torch.float32)
        variance = torch.ones([5, 2], dtype=torch.float32)
        result = torch.nn.functional.gaussian_nll_loss(input=input, target=label, var=variance)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.ones([5, 2], dtype=torch.float32)
        label = torch.ones([5, 2], dtype=torch.float32)
        variance = torch.ones([5, 2], dtype=torch.float32)
        result = torch.nn.functional.gaussian_nll_loss(target=label, var=variance, input=input)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.ones([5, 2], dtype=torch.float32)
        label = torch.ones([5, 2], dtype=torch.float32)
        variance = torch.ones([5, 2], dtype=torch.float32)
        result = torch.nn.functional.gaussian_nll_loss(input=input, target=label, var=variance, full=False, eps=1e-06, reduction='mean')
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.ones([5, 2], dtype=torch.float32)
        label = torch.ones([5, 2], dtype=torch.float32)
        variance = torch.ones([5, 2], dtype=torch.float32)
        result = torch.nn.functional.gaussian_nll_loss(input, label, variance, False, 1e-06, 'mean')
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    """All keyword arguments"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[0.5, -0.3], [1.2, 0.7], [-0.8, 0.4]], dtype=torch.float32)
        label = torch.tensor([[0.4, -0.2], [1.0, 0.5], [-0.6, 0.3]], dtype=torch.float32)
        variance = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=torch.float32)
        result = torch.nn.functional.gaussian_nll_loss(input=input, target=label, var=variance, full=True, eps=1e-08, reduction='sum')
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    """Mixed positional and keyword arguments"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[0.5, -0.3], [1.2, 0.7]], dtype=torch.float32)
        label = torch.tensor([[0.4, -0.2], [1.0, 0.5]], dtype=torch.float32)
        variance = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32)
        result = torch.nn.functional.gaussian_nll_loss(input, label, variance, full=True, reduction='none')
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    """Out-of-order keyword arguments"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[0.5, -0.3, 0.1], [1.2, 0.7, -0.5]], dtype=torch.float32)
        label = torch.tensor([[0.4, -0.2, 0.0], [1.0, 0.5, -0.4]], dtype=torch.float32)
        variance = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=torch.float32)
        result = torch.nn.functional.gaussian_nll_loss(reduction='mean', eps=1e-06, full=False, var=variance, target=label, input=input)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    """3D input tensors"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.ones([2, 3, 4], dtype=torch.float32) * 0.5
        label = torch.ones([2, 3, 4], dtype=torch.float32)
        variance = torch.ones([2, 3, 4], dtype=torch.float32) * 2.0
        result = torch.nn.functional.gaussian_nll_loss(input, label, variance)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_10():
    """float64 input"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[0.5, -0.3], [1.2, 0.7]], dtype=torch.float64)
        label = torch.tensor([[0.4, -0.2], [1.0, 0.5]], dtype=torch.float64)
        variance = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float64)
        result = torch.nn.functional.gaussian_nll_loss(input, label, variance, reduction='sum')
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_11():
    """Variable args unpacking"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.ones([5, 2], dtype=torch.float32)
        label = torch.ones([5, 2], dtype=torch.float32)
        variance = torch.ones([5, 2], dtype=torch.float32)
        args = (input, label, variance)
        result = torch.nn.functional.gaussian_nll_loss(*args)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_12():
    """Default args omitted"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[1.5, 2.3], [-0.5, 1.7]], dtype=torch.float32)
        label = torch.tensor([[1.0, 2.0], [-0.3, 1.5]], dtype=torch.float32)
        variance = torch.tensor([[0.2, 0.3], [0.4, 0.5]], dtype=torch.float32)
        result = torch.nn.functional.gaussian_nll_loss(input, label, variance)
        """
    )
    obj.run(pytorch_code, ["result"])
