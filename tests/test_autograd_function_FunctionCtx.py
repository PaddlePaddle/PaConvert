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

obj = APIBase("torch.autograd.function.FunctionCtx")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torch.autograd import Function

        # Inherit from Function
        class MyFunction(Function):
            @staticmethod
            def forward(ctx, x):
                # Store some tensors for backward
                ctx.x = x
                return x * 2

            @staticmethod
            def backward(ctx, grad_output):
                # Retrieve stored tensors
                x = ctx.x
                grad_input = grad_output * 2
                return grad_input

        data = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        output = MyFunction.apply(data)
        output.backward(torch.tensor([1.0, 1.0, 1.0]))

        result = data.grad
        result.requires_grad = False
        """
    )
    obj.run(pytorch_code, ["result"])
