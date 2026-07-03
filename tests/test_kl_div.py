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

obj = APIBase("torch.kl_div")


def test_case_1():
    """Basic usage with default reduction - NOT SUPPORTED: torch.kl_div requires int reduction, paddle expects string"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.arange(0, 15, dtype=torch.float32, requires_grad=True).reshape((3, 5))
        target = torch.arange(100, 160, 4, dtype=torch.float32, requires_grad=True).reshape((3, 5)) + 4
        result = torch.kl_div(input, target)
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        unsupport=True,
        reason="torch.kl_div and paddle.kl_div have incompatible reduction parameter types (int vs string) and different default reduction behaviors under ChangePrefixMatcher",
    )


def test_case_2():
    """Positional arguments test - NOT SUPPORTED: torch.kl_div requires int reduction, paddle expects string"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.arange(0, 15, dtype=torch.float32, requires_grad=True).reshape((3, 5))
        target = torch.arange(100, 160, 4, dtype=torch.float32, requires_grad=True).reshape((3, 5)) + 4
        result = torch.kl_div(input, target, "mean")
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        unsupport=True,
        reason="torch.kl_div and paddle.kl_div have incompatible reduction parameter types (int vs string) and different default reduction behaviors under ChangePrefixMatcher",
    )


def test_case_3():
    """Keyword arguments test - NOT SUPPORTED: torch.kl_div requires int reduction, paddle expects string"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.arange(0, 15, dtype=torch.float32, requires_grad=True).reshape((3, 5))
        target = torch.arange(100, 160, 4, dtype=torch.float32, requires_grad=True).reshape((3, 5)) + 4
        result = torch.kl_div(input=input, target=target, reduction="sum")
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        unsupport=True,
        reason="torch.kl_div and paddle.kl_div have incompatible reduction parameter types (int vs string) and different default reduction behaviors under ChangePrefixMatcher",
    )


def test_case_4():
    """Keyword arguments out of order test - NOT SUPPORTED: torch.kl_div requires int reduction, paddle expects string"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.arange(0, 15, dtype=torch.float32, requires_grad=True).reshape((3, 5))
        target = torch.arange(100, 160, 4, dtype=torch.float32, requires_grad=True).reshape((3, 5)) + 4
        result = torch.kl_div(reduction="sum", input=input, target=target)
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        unsupport=True,
        reason="torch.kl_div and paddle.kl_div have incompatible reduction parameter types (int vs string) and different default reduction behaviors under ChangePrefixMatcher",
    )


def test_case_5():
    """Gradient computation test - NOT SUPPORTED: torch.kl_div requires int reduction, paddle expects string"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.arange(0, 15, dtype=torch.float32, requires_grad=True).reshape((3, 5))
        target = torch.arange(100, 160, 4, dtype=torch.float32, requires_grad=True).reshape((3, 5)) + 4
        y = torch.kl_div(input, target)
        y.sum().backward()
        x_grad = input.grad
        """
    )
    obj.run(
        pytorch_code,
        ["y", "x_grad"],
        check_stop_gradient=False,
        unsupport=True,
        reason="torch.kl_div and paddle.kl_div have incompatible reduction parameter types (int vs string) and different default reduction behaviors under ChangePrefixMatcher",
    )


def test_case_6():
    """reduction='none' test - NOT SUPPORTED: torch.kl_div requires int reduction, paddle expects string"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.arange(0, 15, dtype=torch.float32, requires_grad=True).reshape((3, 5))
        target = torch.arange(100, 160, 4, dtype=torch.float32, requires_grad=True).reshape((3, 5)) + 4
        result = torch.kl_div(input, target, "none")
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        unsupport=True,
        reason="torch.kl_div and paddle.kl_div have incompatible reduction parameter types (int vs string) and different default reduction behaviors under ChangePrefixMatcher",
    )


def test_case_7():
    """log_target=True test - NOT SUPPORTED: torch.kl_div requires int reduction, paddle expects string"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.arange(0, 15, dtype=torch.float32, requires_grad=True).reshape((3, 5))
        target = torch.arange(100, 160, 4, dtype=torch.float32, requires_grad=True).reshape((3, 5)) + 4
        result = torch.kl_div(input, target, log_target=True)
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        unsupport=True,
        reason="torch.kl_div and paddle.kl_div have incompatible reduction parameter types (int vs string) and different default reduction behaviors under ChangePrefixMatcher",
    )


def test_case_8():
    """reduction='mean' with log_target=True - NOT SUPPORTED: torch.kl_div requires int reduction, paddle expects string"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.arange(0, 15, dtype=torch.float32, requires_grad=True).reshape((3, 5))
        target = torch.arange(100, 160, 4, dtype=torch.float32, requires_grad=True).reshape((3, 5)) + 4
        result = torch.kl_div(input, target, reduction="mean", log_target=True)
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        unsupport=True,
        reason="torch.kl_div and paddle.kl_div have incompatible reduction parameter types (int vs string) and different default reduction behaviors under ChangePrefixMatcher",
    )


def test_case_9():
    """Mixed positional and keyword arguments - NOT SUPPORTED: torch.kl_div requires int reduction, paddle expects string"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.arange(0, 15, dtype=torch.float32, requires_grad=True).reshape((3, 5))
        target = torch.arange(100, 160, 4, dtype=torch.float32, requires_grad=True).reshape((3, 5)) + 4
        result = torch.kl_div(input, target, reduction="sum", log_target=True)
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        unsupport=True,
        reason="torch.kl_div and paddle.kl_div have incompatible reduction parameter types (int vs string) and different default reduction behaviors under ChangePrefixMatcher",
    )


def test_case_10():
    """Edge case test with 1D tensors - NOT SUPPORTED: torch.kl_div requires int reduction, paddle expects string"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        target = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)
        result = torch.kl_div(input, target)
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        unsupport=True,
        reason="torch.kl_div and paddle.kl_div have incompatible reduction parameter types (int vs string) and different default reduction behaviors under ChangePrefixMatcher",
    )
