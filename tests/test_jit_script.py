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

obj = APIBase("torch.jit.script")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        @torch.jit.script
        def foo(x, scale, shift):
            return shift + scale * x
        x = torch.tensor([-1, -2, 3.0])
        result = foo(x,x,x)
        """
    )
    paddle_code = textwrap.dedent(
        """
        import paddle


        @paddle.jit.to_static
        def foo(x, scale, shift):
            return shift + scale * x


        x = paddle.tensor([-1, -2, 3.0])
        result = foo(x, x, x)
        """
    )
    obj.run(pytorch_code, expect_paddle_code=paddle_code)


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        def add(x, y):
            return x + y


        scripted_add = torch.jit.script(add)

        x = torch.tensor(1)
        y = torch.tensor(2)
        result = scripted_add(x, y)
        """
    )
    paddle_code = textwrap.dedent(
        """
        import paddle


        def add(x, y):
            return x + y


        scripted_add = paddle.jit.to_static(function=add)
        x = paddle.tensor(1)
        y = paddle.tensor(2)
        result = scripted_add(x, y)
        """
    )
    obj.run(pytorch_code, expect_paddle_code=paddle_code)


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        def add(x, y):
            return x + y


        scripted_add = torch.jit.script(obj=add, optimize=True, example_inputs=[torch.tensor([0]), torch.tensor([2])])

        x = torch.tensor(1)
        y = torch.tensor(2)
        result = scripted_add(x, y)
        """
    )
    paddle_code = textwrap.dedent(
        """
        import paddle


        def add(x, y):
            return x + y


        scripted_add = paddle.jit.to_static(
            function=add, input_spec=[paddle.tensor([0]), paddle.tensor([2])]
        )
        x = paddle.tensor(1)
        y = paddle.tensor(2)
        result = scripted_add(x, y)
        """
    )
    obj.run(pytorch_code, expect_paddle_code=paddle_code)
