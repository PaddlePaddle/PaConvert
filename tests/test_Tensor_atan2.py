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
#

import textwrap

from apibase import APIBase

obj = APIBase("torch.Tensor.atan2")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([ 0.9041,  0.0196, -0.3108, -2.4423])
        other = torch.tensor([ 0.2341,  0.2539, -0.6256, -0.6448])
        result = input.atan2(other)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.tensor([ 0.9041,  0.0196, -0.3108, -2.4423]).atan2(torch.tensor([ 0.2341,  0.2539, -0.6256, -0.6448]))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([ 0.9041,  0.0196, -0.3108, -2.4423])
        other = torch.tensor([ 0.2341,  0.2539, -0.6256, -0.6448])
        result = input.atan2(other=other)
        """
    )
    obj.run(pytorch_code, ["result"])


# Test with float32 dtype
def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([0.9041, 0.0196, -0.3108, -2.4423], dtype=torch.float32)
        other = torch.tensor([0.2341, 0.2539, -0.6256, -0.6448], dtype=torch.float32)
        result = input.atan2(other)
        """
    )
    obj.run(pytorch_code, ["result"])


# Test with float64 dtype
def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([0.9041, 0.0196, -0.3108, -2.4423], dtype=torch.float64)
        other = torch.tensor([0.2341, 0.2539, -0.6256, -0.6448], dtype=torch.float64)
        result = input.atan2(other)
        """
    )
    obj.run(pytorch_code, ["result"])


# Test with 2D input
def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[0.9041, 0.0196], [-0.3108, -2.4423]])
        other = torch.tensor([[0.2341, 0.2539], [-0.6256, -0.6448]])
        result = input.atan2(other)
        """
    )
    obj.run(pytorch_code, ["result"])


# Test with 3D input
def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[[0.9041, 0.0196], [-0.3108, -2.4423]],
                              [[0.5012, -0.1234], [0.7891, 0.3456]]])
        other = torch.tensor([[[0.2341, 0.2539], [-0.6256, -0.6448]],
                              [[0.1122, -0.3344], [0.5566, 0.7788]]])
        result = input.atan2(other)
        """
    )
    obj.run(pytorch_code, ["result"])


# Test with method chaining
def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([0.9041, 0.0196, -0.3108, -2.4423])
        other = torch.tensor([0.2341, 0.2539, -0.6256, -0.6448])
        result = input.clone().atan2(other)
        """
    )
    obj.run(pytorch_code, ["result"])


# Test with broadcasting (different shapes)
def test_case_9():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[0.9041], [0.0196], [-0.3108]])
        other = torch.tensor([0.2341, 0.2539, -0.6256])
        result = input.atan2(other)
        """
    )
    obj.run(pytorch_code, ["result"])


# Test with 2D input and keyword argument
def test_case_10():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[0.9041, 0.0196], [-0.3108, -2.4423]])
        other = torch.tensor([[0.2341, 0.2539], [-0.6256, -0.6448]])
        result = input.atan2(other=other)
        """
    )
    obj.run(pytorch_code, ["result"])
