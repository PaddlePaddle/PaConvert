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
import textwrap

from apibase import APIBase

obj = APIBase("torch._assert")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        torch._assert(1==1, "not equal")
        """
    )
    obj.run(pytorch_code)


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        torch._assert(condition=(1==1), message="not equal")
        """
    )
    obj.run(pytorch_code)


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        torch._assert((1==1), message="not equal")
        """
    )
    obj.run(pytorch_code)


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        torch._assert(message="not equal", condition=(1==1))
        """
    )
    obj.run(pytorch_code)


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        _, _, H, W = [1, 2, 3, 4]
        img_size = [3, 4]
        torch._assert(
            H == img_size[0],
            f"Input image height ({H}) doesn't match model ({img_size[0]}).",
        )
        """
    )
    obj.run(pytorch_code)


def test_case_6():
    """Tensor condition"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([True])
        torch._assert(x, "tensor should be true")
        """
    )
    obj.run(pytorch_code)


def test_case_7():
    """Boolean literal condition"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        torch._assert(True, "literal true")
        """
    )
    obj.run(pytorch_code)


def test_case_8():
    """Integer as condition (truthy)"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        torch._assert(1, "one is truthy")
        """
    )
    obj.run(pytorch_code)


def test_case_9():
    """Empty string message"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        torch._assert(True, "")
        """
    )
    obj.run(pytorch_code)


def test_case_10():
    """Variable as condition"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        cond = (2 > 1)
        msg = "two is greater"
        torch._assert(cond, msg)
        """
    )
    obj.run(pytorch_code)


def test_case_11():
    """Comparison expression with len()"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        data = [1, 2, 3]
        torch._assert(len(data) == 3, "length mismatch")
        """
    )
    obj.run(pytorch_code)


def test_case_12():
    """Tensor comparison condition"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor(5.0)
        torch._assert(x > 0, "x should be positive")
        """
    )
    obj.run(pytorch_code)
