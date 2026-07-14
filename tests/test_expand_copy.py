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

obj = APIBase("torch.expand_copy")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1, 2, 3])
        result = torch.expand_copy(x, (3, 3))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1, 2, 3])
        result = torch.expand_copy(x, size=(3, 3))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1], [3], [4]])
        result = torch.expand_copy(input=x, size=(3, 3))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1], [3], [2]])
        result = torch.expand_copy(size=(3, 3), input=x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1, 2, 3])
        size = (3, 3)
        result = torch.expand_copy(x, size)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1, 2, 3])
        result = torch.expand_copy(x, (3, -1))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1], [3], [4]])
        result = torch.expand_copy(x, [3, 3])
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1, 2, 3])
        result = torch.expand_copy(input=x, size=(3, 3))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1, 2, 3])
        result = torch.expand_copy(x, (2, 3))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_10():
    """Gradient computation test"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([0.5, 1.0, 2.0], requires_grad=True)
        y = torch.expand_copy(a, (2, 3))
        y.sum().backward()
        a_grad = a.grad
        """
    )
    obj.run(pytorch_code, ["y", "a_grad"], check_stop_gradient=False)


def test_case_11():
    """Expression argument test"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.expand_copy(torch.tensor([1.0, 2.0, 3.0]) + torch.tensor([0.5, 0.5, 0.5]), (2, 3))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_12():
    """2D tensor test"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[1.4309, 1.2706], [-0.8562, 0.9796]])
        result = torch.expand_copy(a, (2, 2))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_13():
    """3D tensor test"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]])
        result = torch.expand_copy(a, (2, 2, 2))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_14():
    """float64 dtype test"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([1.4309, 1.2706], dtype=torch.float64)
        result = torch.expand_copy(a, (2, 2))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_15():
    """Expression argument test"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.expand_copy(torch.tensor([1.0, 2.0, 3.0]) + torch.tensor([0.5, 0.5, 0.5]), (2, 3))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_16():
    """Variable arguments"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1, 2, 3])
        args = (x, (3, 3))
        result = torch.expand_copy(*args)
        """
    )
    obj.run(pytorch_code, ["result"])
