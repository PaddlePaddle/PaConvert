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

obj = APIBase("torch.max")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 2, 3], [3, 4, 6]])
        out = torch.tensor([1, 2, 3])
        result = torch.max(x)
        """
    )
    obj.run(pytorch_code, ["result", "out"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.max(torch.tensor([[1, 2, 3], [3, 4, 6]]))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 2, 3], [3, 4, 6]])
        result = torch.max(x, 1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 2, 3], [3, 4, 6]])
        result = torch.max(x, -1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 2, 3], [3, 4, 6]])
        result = torch.max(x, dim=-1, keepdim=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 2, 3], [3, 4, 6]])
        result = torch.max(input=x, dim=1, keepdim=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 2, 1], [3, 4, 6]])
        out = [torch.tensor(0), torch.tensor(1)]
        result = torch.max(x, 1, False, out=out)
        """
    )
    obj.run(pytorch_code, ["result", "out"])


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        dim = 1
        keepdim = False
        result = torch.max(torch.tensor([[1, 2, 3], [3, 4, 6]]), dim, keepdim)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.max(torch.tensor([[1, 2, 3], [3, 4, 6]]), torch.tensor([[1, 0, 3], [3, 4, 3]]))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_10():
    pytorch_code = textwrap.dedent(
        """
        import torch
        other = torch.tensor([[1, 0, 3], [3, 4, 3]])
        result = torch.max(torch.tensor([[1, 2, 3], [3, 4, 6]]), other)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_11():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.max(input=torch.tensor([[1, 2, 3], [3, 4, 6]]), other=torch.tensor([[1, 0, 3], [3, 4, 3]]))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_12():
    pytorch_code = textwrap.dedent(
        """
        import torch
        other = torch.tensor([[1, 0, 3], [3, 4, 3]])
        out = torch.tensor([[1, 0, 3], [3, 4, 3]])
        result = torch.max(input=torch.tensor([[1, 2, 3], [3, 4, 6]]), other=other, out=out)
        """
    )
    obj.run(pytorch_code, ["result", "out"])


def test_case_13():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 2, 1], [3, 4, 6]])
        out = [torch.tensor(0), torch.tensor(1)]
        result = torch.max(input=x, dim=1, keepdim=False, out=out)
        """
    )
    obj.run(pytorch_code, ["result", "out"])


def test_case_14():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 2, 1], [3, 4, 6]])
        out = [torch.tensor(0), torch.tensor(1)]
        result = torch.max(input=x, keepdim=False, dim=1, out=out)
        """
    )
    obj.run(pytorch_code, ["result", "out"])


def test_case_15():
    pytorch_code = textwrap.dedent(
        """
        import torch
        other = torch.tensor([[1, 0, 3], [3, 4, 3]])
        out = torch.tensor([[1, 0, 3], [3, 4, 3]])
        result = torch.max(other=other, out=out, input=torch.tensor([[1, 2, 3], [3, 4, 6]]))
        """
    )
    obj.run(pytorch_code, ["result", "out"])


def test_case_16():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 2, 1], [3, 4, 6]], dtype=torch.float32)
        x.requires_grad = True
        y = x * x + x
        values, indices = torch.max(keepdim=False, dim=1, input=y)
        values.backward(torch.ones_like(values))
        grad_tensor = x.grad
        grad_tensor.requires_grad = False
        """
    )
    obj.run(pytorch_code, ["indices", "grad_tensor"])


def test_case_17():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 2, 1], [3, 4, 6]], dtype=torch.float32)
        x.requires_grad = True
        y = x * 2
        values = torch.max(input=y)
        values.backward()
        grad_tensor = x.grad
        grad_tensor.requires_grad = False
        """
    )
    obj.run(pytorch_code, ["values", "grad_tensor"])


def test_case_18():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 1, 2, 2], [3, 3, 4, 4], [1, 3, 2, 4]], dtype=torch.float32)
        x.requires_grad = True
        y = x * 2 + 1
        results = torch.max(input=y, dim = 0)
        results.values.backward(torch.ones_like(results.values))
        grad_tensor = x.grad
        grad_tensor.requires_grad = False
        """
    )
    obj.run(pytorch_code, ["results", "grad_tensor"])


def test_case_19():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 1, 2, 2], [3, 3, 4, 4], [1, 3, 2, 4]], dtype=torch.float32)
        y = torch.tensor([[2, 1, 1, 2], [4, 3, 4, 3], [3, 1, 4, 2]], dtype=torch.float32)
        x.requires_grad = True
        y.requires_grad = True
        result = torch.max(x, y)
        result.backward(torch.ones_like(result))
        x.grad.requires_grad = False
        y.grad.requires_grad = False
        x_grad = x.grad
        y_grad = y.grad
        """
    )
    obj.run(pytorch_code, ["result", "x_grad", "y_grad"])
