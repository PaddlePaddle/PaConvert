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

obj = APIBase("torch.autograd.grad_mode.set_grad_enabled")


def test_case_1():
    """Basic usage with True"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        torch.autograd.grad_mode.set_grad_enabled(True)
        result = torch.is_grad_enabled()
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    """Basic usage with False"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        torch.autograd.grad_mode.set_grad_enabled(False)
        result = torch.is_grad_enabled()
        torch.autograd.grad_mode.set_grad_enabled(True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    """Keyword argument"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        torch.autograd.grad_mode.set_grad_enabled(mode=True)
        result = torch.is_grad_enabled()
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    """Keyword argument with False"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        torch.autograd.grad_mode.set_grad_enabled(mode=False)
        result = torch.is_grad_enabled()
        torch.autograd.grad_mode.set_grad_enabled(True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    """Variable argument"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        mode = True
        torch.autograd.grad_mode.set_grad_enabled(mode)
        result = torch.is_grad_enabled()
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    """Toggle off then on"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        torch.autograd.grad_mode.set_grad_enabled(False)
        result1 = torch.is_grad_enabled()
        torch.autograd.grad_mode.set_grad_enabled(True)
        result2 = torch.is_grad_enabled()
        result = (result1, result2)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    """Expression mode argument"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        torch.autograd.grad_mode.set_grad_enabled(1 == 0)
        result = torch.is_grad_enabled()
        torch.autograd.grad_mode.set_grad_enabled(True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    """Gradient computation test"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([0.5, 1.0, 2.0], requires_grad=True)
        y = torch.autograd.grad_mode.set_grad_enabled(True)
        y = torch.is_grad_enabled()
        a_grad = a.grad
        """
    )
    obj.run(pytorch_code, ["y", "a_grad"], check_stop_gradient=False)
