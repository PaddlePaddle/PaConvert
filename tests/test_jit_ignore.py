# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

obj = APIBase("torch.jit.ignore")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn


        class MyModule(nn.Module):
            def forward(self, x):
                return x * 2

            @torch.jit.ignore
            def helper_function(self, x):
                return x + 10


        model = torch.jit.script(MyModule())
        result = model(torch.tensor([5.0]))
        """
    )
    paddle_code = textwrap.dedent(
        """
        import paddle


        class MyModule(paddle.nn.Layer):
            def forward(self, x):
                return x * 2

            @paddle.jit.not_to_static
            def helper_function(self, x):
                return x + 10


        model = paddle.jit.to_static(function=MyModule())
        result = model(paddle.to_tensor(data=[5.0]))
        """
    )
    obj.run(pytorch_code, expect_paddle_code=paddle_code)


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn


        class MyModule(nn.Module):
            def forward(self, x):
                return x * 2

            @torch.jit.ignore(drop=True)
            def helper_function(self, x):
                return x + 10


        model = torch.jit.script(MyModule())
        result = model(torch.tensor([5.0]))
        """
    )
    obj.run(pytorch_code, unsupport=True, reason="Not support parameter dropout")
