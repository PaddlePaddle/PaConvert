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

obj = APIBase("torch.autograd.graph.saved_tensors_hooks")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        def pack_hook(x):
            # print("Packing", x)
            return x.numpy()

        def unpack_hook(x):
            # print("UnPacking", x)
            return torch.tensor(x)

        a = torch.ones([3,3])
        b = torch.ones([3,3]) * 2
        a.requires_grad = True
        b.requires_grad = True
        with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
            y = torch.multiply(a, b)
        y.sum().backward()
        result = y
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torch.autograd import Function

        class cus_tanh(Function):
            @staticmethod
            def forward(ctx, a, b):
                y = torch.multiply(a, b)
                ctx.save_for_backward(a, b)
                return y

            @staticmethod
            def backward(ctx, dy):
                grad_a = dy * 2
                grad_b = dy * 3
                return grad_a, grad_b

        def pack_hook(x):
            # print("Packing", x)
            return x.numpy()

        def unpack_hook(x):
            # print("UnPacking", x)
            return torch.tensor(x)

        a = torch.ones([3,3])
        b = torch.ones([3,3]) * 2
        a.requires_grad = True
        b.requires_grad = True
        with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
            y = cus_tanh.apply(a, b)
        y.sum().backward()
        result = y
        """
    )
    obj.run(pytorch_code, ["result"])
