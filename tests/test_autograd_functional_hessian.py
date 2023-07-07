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

obj = APIBase("torch.autograd.functional.hessian")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        def func(x):
            return torch.sum(x * x)
        x = torch.rand(2, 2)
        h = torch.autograd.functional.hessian(func, x)
        result = h[:]
        result.requires_grad = False
        result = torch.flatten(result)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch

        def func(x):
            return 2 * torch.sum(x * x + 3 * x)

        x = torch.rand(2, 2)
        h = torch.autograd.functional.hessian(func, x)
        result = h[:]
        result.requires_grad = False
        result = torch.flatten(result)
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
        h = torch.autograd.functional.hessian(func, x)
        result = h[:]
        result.requires_grad = False
        result = torch.flatten(result)
        """
    )
    obj.run(pytorch_code, ["result"])
