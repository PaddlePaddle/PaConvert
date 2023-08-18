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

obj = APIBase("torch.autograd.Function.backward")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torch.autograd import Function

        # Inherit from Function
        class cus_tanh(Function):
            @staticmethod
            def forward(ctx, x, func=torch.square):
                ctx.func = func
                y = func(x)
                return y

            @staticmethod
            def backward(ctx, dy):
                grad = dy + 1
                return grad

        data = torch.ones([2, 3], dtype=torch.float64)
        data.requires_grad = True
        z = cus_tanh.apply(data)
        z.mean().backward()

        result = data.grad
        """
    )
    obj.run(pytorch_code, ["result"], check_stop_gradient=False)
