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

obj = APIBase("torch.autograd.functional.jvp")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        def func(x):
            return torch.sum(x * x)
        x = torch.ones(2, 3)
        v = torch.ones(2, 3)
        h = torch.autograd.functional.jvp(func, x, v)
        result = h[:]
        for item in result:
            item.requires_grad = False
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch

        def func(x):
            return 2 * torch.sum(x * x + 3 * x)

        x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        v = torch.ones(2, 3)
        h = torch.autograd.functional.jvp(func, x, v)
        result = h[:]
        for item in result:
            item.requires_grad = False
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch

        def func(x):
            return torch.sum(x)

        x = torch.tensor([1.0, 2.0])
        v = torch.ones(2)
        h = torch.autograd.functional.jvp(func, x, v)
        result = h[:]
        for item in result:
            item.requires_grad = False
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch

        def func(x):
            return torch.sum(x)

        x = torch.tensor([1.0, 2.0])
        v = torch.ones(2)
        h = torch.autograd.functional.jvp(func, x, v, create_graph=True)
        result = h[:]
        for item in result:
            item.requires_grad = False
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
            return torch.sum(x)

        x = torch.tensor([1.0, 2.0])
        v = torch.ones(2)
        h = torch.autograd.functional.jvp(func, x, v, strict=False)
        result = h[:]
        for item in result:
            item.requires_grad = False
        """
    )
    obj.run(pytorch_code, ["result"], unsupport=True, reason="paddle unsupport strict")


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch

        def func(x):
            return torch.sum(x)

        x = torch.tensor([1.0, 2.0])
        v = torch.ones(2)
        h = torch.autograd.functional.jvp(func=func, inputs=x, v=v)
        result = h[:]
        for item in result:
            item.requires_grad = False
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch

        def func(x):
            return torch.sum(x)

        x = torch.tensor([1.0, 2.0])
        v = torch.ones(2)
        h = torch.autograd.functional.jvp(v=v, inputs=x, func=func)
        result = h[:]
        for item in result:
            item.requires_grad = False
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        def func(x):
            return torch.sum(x * x)
        x = torch.tensor(1.)
        h = torch.autograd.functional.jvp(func, x)
        result = h[:]
        for item in result:
            item.requires_grad = False
        """
    )
    obj.run(pytorch_code, ["result"])
