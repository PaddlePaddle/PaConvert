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

obj = APIBase("torch.special.logit")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([0.2796, 0.9331, 0.6486, 0.1523, 0.6516])
        result = torch.special.logit(input, eps=1e-6)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([0.2796, 0.9331, 0.6486, 0.1523, 0.6516])
        eps = 1e-6
        result = torch.special.logit(input, eps)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([0.2796, 0.9331, 0.6486, 0.1523, 0.6516])
        result = torch.special.logit(x, eps=1e-6)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([0.2796, 0.9331, 0.6486, 0.1523, 0.6516])
        out = torch.zeros(5)
        result = torch.special.logit(input=x, eps=1e-6, out=out)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([0.2796, 0.9331, 0.6486, 0.1523, 0.6516])
        out = torch.zeros(5)
        result = torch.special.logit(eps=1e-6, input=x, out=out)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([0.2796, 0.9331, 0.6486, 0.1523, 0.6516])
        result = torch.special.logit(input)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    """No eps, keyword input="""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([0.2796, 0.9331, 0.6486, 0.1523, 0.6516])
        result = torch.special.logit(input=input)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    """2D input with eps"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[0.2, 0.5, 0.8], [0.1, 0.9, 0.3]])
        result = torch.special.logit(input, eps=1e-6)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    """3D input, no eps"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[[0.2, 0.8], [0.4, 0.6]], [[0.1, 0.9], [0.3, 0.7]]])
        result = torch.special.logit(input)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_10():
    """float64 dtype"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([0.2796, 0.9331, 0.6486], dtype=torch.float64)
        result = torch.special.logit(input, eps=1e-6)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_11():
    """out parameter without eps"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([0.2796, 0.9331, 0.6486, 0.1523, 0.6516])
        out = torch.zeros(5)
        result = torch.special.logit(input, out=out)
        """
    )
    obj.run(pytorch_code, ["result", "out"])


def test_case_12():
    """Reordered kwargs: out, input, eps"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([0.2796, 0.9331, 0.6486, 0.1523, 0.6516])
        out = torch.zeros(5)
        result = torch.special.logit(out=out, input=input, eps=1e-6)
        """
    )
    obj.run(pytorch_code, ["result", "out"])


def test_case_13():
    """Variable unpacking"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        args = (torch.tensor([0.2796, 0.9331, 0.6486, 0.1523, 0.6516]), 1e-6)
        result = torch.special.logit(*args)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_14():
    """Expression as eps argument"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([0.2796, 0.9331, 0.6486, 0.1523, 0.6516])
        result = torch.special.logit(input, 1e-3 * 1e-3)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_15():
    """Gradient computation"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([0.2796, 0.9331, 0.6486], requires_grad=True)
        y = torch.special.logit(x, eps=1e-6)
        y.sum().backward()
        x_grad = x.grad
        """
    )
    obj.run(pytorch_code, ["y", "x_grad"], check_stop_gradient=False)
