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

obj = APIBase("torch.autograd.function.Function")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = isinstance(torch.autograd.function.Function, type)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.autograd.function.Function is torch.autograd.Function
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        class MyReLU(torch.autograd.function.Function):
            @staticmethod
            def forward(ctx, input):
                ctx.save_for_backward(input)
                return input.clamp(min=0)
            @staticmethod
            def backward(ctx, grad_output):
                input, = ctx.saved_tensors
                grad_input = grad_output.clone()
                grad_input[input < 0] = 0
                return grad_input
        result = MyReLU.apply(torch.tensor([-1.0, 0.0, 1.0]))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.autograd.function.Function.__name__
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = issubclass(torch.autograd.function.Function, type)
        """
    )
    obj.run(pytorch_code, ["result"])
