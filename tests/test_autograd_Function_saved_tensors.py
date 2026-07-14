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

obj = APIBase("torch.autograd.Function.saved_tensors")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torch.autograd import Function

        class cus_tanh(Function):
            @staticmethod
            def forward(ctx, x):
                y = torch.tanh(x)
                ctx.save_for_backward(x, y)
                return y

            @staticmethod
            def backward(ctx, dy):
                x, y = ctx.saved_tensors
                grad = y + dy + 1
                return grad

        data = torch.ones([2, 3], dtype=torch.float64, requires_grad=True)
        z = cus_tanh.apply(data)
        z.sum().backward()

        result = data.grad
        """
    )
    obj.run(pytorch_code, ["z", "result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torch.autograd import Function

        class cus_tanh(Function):
            @staticmethod
            def forward(ctx, x):
                y = torch.tanh(x)
                ctx.save_for_backward(y)
                return y

            @staticmethod
            def backward(ctx, dy):
                y = ctx.saved_tensors[0]
                grad = y + dy + 1
                return grad

        data = torch.ones([2, 3], dtype=torch.float64, requires_grad=True)
        z = cus_tanh.apply(data)
        z.sum().backward()

        result = data.grad
        """
    )
    obj.run(pytorch_code, ["z", "result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torch.autograd import Function

        class square(Function):
            @staticmethod
            def forward(ctx, x, y):
                ctx.save_for_backward(x, y)
                return x * y

            @staticmethod
            def backward(ctx, grad):
                x, y = ctx.saved_tensors
                return grad * y, grad * x

        a = torch.tensor([2.0, 3.0], dtype=torch.float64, requires_grad=True)
        b = torch.tensor([4.0, 5.0], dtype=torch.float64, requires_grad=True)
        z = square.apply(a, b)
        z.sum().backward()
        result = a.grad
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torch.autograd import Function

        class cube(Function):
            @staticmethod
            def forward(ctx, x):
                result = x ** 3
                ctx.save_for_backward(x)
                return result

            @staticmethod
            def backward(ctx, grad):
                x, = ctx.saved_tensors
                return grad * 3 * x ** 2

        a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True)
        z = cube.apply(a)
        z.sum().backward()
        result = a.grad
        """
    )
    obj.run(pytorch_code, ["result"])
