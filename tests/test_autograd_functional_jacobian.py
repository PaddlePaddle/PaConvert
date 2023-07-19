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

obj = APIBase("torch.autograd.functional.jacobian")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch

        def func(x):
            return x * x

        x = torch.tensor([1., 2.])
        J = torch.autograd.functional.jacobian(func, x)
        result = J[:]
        result.requires_grad = False
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch

        def func(x):
            return torch.cos(x)

        x = torch.tensor([1., 2.])
        J = torch.autograd.functional.jacobian(func, x)
        result = J[:]
        result.requires_grad = False
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch

        def func(x):
            return torch.log(x)

        x = torch.tensor([1., 2.])
        J = torch.autograd.functional.jacobian(func, x)
        result = J[:]
        result.requires_grad = False
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch

        def func(x):
            return torch.log(x)

        x = torch.tensor([1., 2.])
        J = torch.autograd.functional.jacobian(func, x, create_graph=True)
        result = J[:]
        result.requires_grad = False
        """
    )
    obj.run(
        pytorch_code, ["result"], unsupport=True, reason="paddle unsupport create_graph"
    )


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch

        def func(x):
            return torch.log(x)

        x = torch.tensor([1., 2.])
        J = torch.autograd.functional.jacobian(func, x, strict=False)
        result = J[:]
        result.requires_grad = False
        """
    )
    obj.run(pytorch_code, ["result"], unsupport=True, reason="paddle unsupport strict")
