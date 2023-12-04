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

obj = APIBase("torch.autograd.backward")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32, requires_grad=True)
        y = torch.tensor([[3, 2], [3, 4]], dtype=torch.float32)

        grad_tensor1 = torch.tensor([[1,2], [2, 3]], dtype=torch.float32)
        grad_tensor2 = torch.tensor([[1,1], [1, 1]], dtype=torch.float32)

        z1 = torch.matmul(x, y)
        z2 = torch.matmul(x, y)

        torch.autograd.backward([z1, z2], [grad_tensor1, grad_tensor2], True)
        x.grad.requires_grad=False
        result = x.grad
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32, requires_grad=True)
        y = torch.tensor([[3, 2], [3, 4]], dtype=torch.float32)

        grad_tensor1 = torch.tensor([[1,2], [2, 3]], dtype=torch.float32)
        grad_tensor2 = torch.tensor([[1,1], [1, 1]], dtype=torch.float32)

        z1 = torch.matmul(x, y)
        z2 = torch.matmul(x, y)

        torch.autograd.backward([z1, z2], [grad_tensor1, grad_tensor2], retain_graph=False)
        x.grad.requires_grad=False
        result = x.grad
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32, requires_grad=True)
        z1 = x.sum()

        torch.autograd.backward([z1])
        x.grad.requires_grad=False
        result = x.grad
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32, requires_grad=True)
        y = torch.tensor([[3, 2], [3, 4]], dtype=torch.float32)

        grad_tensor1 = torch.tensor([[1,2], [2, 3]], dtype=torch.float32)
        grad_tensor2 = torch.tensor([[1,1], [1, 1]], dtype=torch.float32)

        z1 = torch.matmul(x, y)
        z2 = torch.matmul(x, y)

        torch.autograd.backward(tensors=[z1, z2], grad_tensors=[grad_tensor1, grad_tensor2], retain_graph=False)
        x.grad.requires_grad=False
        result = x.grad
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32, requires_grad=True)
        y = torch.tensor([[3, 2], [3, 4]], dtype=torch.float32)

        grad_tensor1 = torch.tensor([[1,2], [2, 3]], dtype=torch.float32)
        grad_tensor2 = torch.tensor([[1,1], [1, 1]], dtype=torch.float32)

        z1 = torch.matmul(x, y)
        z2 = torch.matmul(x, y)

        torch.autograd.backward(grad_tensors=[grad_tensor1, grad_tensor2], tensors=[z1, z2], retain_graph=False)
        x.grad.requires_grad=False
        result = x.grad
        """
    )
    obj.run(pytorch_code, ["result"])
