# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

obj = APIBase("torch.Tensor.geometric_")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.tensor([-0.6341, -1.4208, -1.0900,  0.5826]).geometric_(0.5)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([-0.6341, -1.4208, -1.0900,  0.5826])
        result = input.geometric_(0.5)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([-0.6341, -1.4208, -1.0900,  0.5826])
        result = input.geometric_(p=0.5)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([-0.6341, -1.4208, -1.0900,  0.5826])
        result = input.geometric_(p=0.5, generator=None)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([-0.6341, -1.4208, -1.0900,  0.5826])
        result = input.geometric_(generator=None, p=0.5)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_6():
    """2D tensor test"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[-0.6341, -1.4208], [-1.0900, 0.5826]])
        result = input.geometric_(0.5)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_7():
    """3D tensor test"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.ones(2, 3, 4)
        result = input.geometric_(0.5)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_8():
    """Float64 dtype test"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([-0.6341, -1.4208, -1.0900, 0.5826], dtype=torch.float64)
        result = input.geometric_(0.5)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_9():
    """Expression argument test"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([-0.6341, -1.4208, -1.0900, 0.5826])
        result = input.geometric_(0.3 + 0.2)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_10():
    """Boundary value test: p = 0.1"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = input.geometric_(0.1)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_11():
    """Boundary value test: p = 0.9"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = input.geometric_(0.9)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_12():
    """In-place modification verification"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([1.0, 2.0, 3.0, 4.0])
        original_id = id(input)
        result = input.geometric_(p=0.5)
        modified_id = id(result)
        same_object = (original_id == modified_id)
        """
    )
    obj.run(pytorch_code, ["result", "same_object"], check_value=False)


def test_case_13():
    """Large tensor test"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.ones(100, 100)
        result = input.geometric_(0.5)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)
