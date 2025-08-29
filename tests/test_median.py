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

obj = APIBase("torch.median")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([1.4907, 1.0593, 1.5696])
        result = torch.median(input)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[1.4907, 1.0593, 1.5696], [1.4907, 1.0593, 1.5696]])
        result = torch.median(input, 1)
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
    )


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[1.4907, 1.0593, 1.5696], [1.4907, 1.0593, 1.5696]])
        result = torch.median(input, 1, keepdim=True)
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
    )


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[1.4907, 1.0593, 1.5696], [1.4907, 1.0593, 1.5696]])
        result = torch.median(input, dim=1, keepdim=True)
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
    )


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[1.4907, 1.0593, 1.5696], [1.4907, 1.0593, 1.5696]])
        out = (torch.tensor([[1.1], [1.2]]), torch.tensor([[1], [2]]))
        result = torch.median(input=input, dim=1, keepdim=True, out=out)
        """
    )
    obj.run(pytorch_code, ["result", "out"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([1, 4, 6])
        result = torch.median(input, 0)
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
    )


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[1.4907, 1.0593, 1.5696], [1.4907, 1.0593, 1.5696]])
        result = torch.median(input, dim=1, keepdim=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[1.4907, 1.0593, 1.5696], [1.4907, 1.0593, 1.5696]])
        result = torch.median(input, 1, True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[1.4907, 1.0593, 1.5696], [1.4907, 1.0593, 1.5696]])
        out = (torch.tensor([[1.1], [1.2]]), torch.tensor([[1], [2]]))
        result = torch.median(keepdim=True, out=out, input=input, dim=1)
        """
    )
    obj.run(pytorch_code, ["result", "out"])


def test_case_10():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32, requires_grad=True)
        y = x * x + x
        values, indices = torch.median(y, dim=1)
        values.backward(torch.ones_like(values))
        grad_tensor = x.grad
        grad_tensor.requires_grad = False
        """
    )
    obj.run(pytorch_code, ["indices", "grad_tensor"])


def test_case_11():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32, requires_grad=True)
        y = x * 3
        values = torch.median(y)
        values.backward()
        grad_tensor = x.grad
        grad_tensor.requires_grad = False
        """
    )
    obj.run(pytorch_code, ["values", "grad_tensor"])


def test_case_12():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=torch.float32, requires_grad=True)
        y = x * 2 + 1
        results = torch.median(y, dim=0)
        results.values.backward(torch.ones_like(results.values))
        grad_tensor = x.grad
        grad_tensor.requires_grad = False
        """
    )
    obj.run(pytorch_code, ["results", "grad_tensor"])


def test_case_13():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32, requires_grad=True)
        y = torch.tensor([[2.0, 1.0, 3.0], [5.0, 4.0, 6.0]], dtype=torch.float32, requires_grad=True)
        result = torch.median(torch.cat((x, y), dim=1), dim=1)
        result.values.backward(torch.ones_like(result.values))
        x.grad.requires_grad = False
        y.grad.requires_grad = False
        x_grad = x.grad
        y_grad = y.grad
        """
    )
    obj.run(pytorch_code, ["result", "x_grad", "y_grad"])
