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

obj = APIBase("torch.jit.save")


# change paddle.base.framework.EagerParamBase.from_tensor to paddle.nn.parameter.Parameter
def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn

        class MyModule(nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.dummy_param = nn.Parameter(torch.tensor([1.0]))

            def forward(self, x):
                return x + 10


        m = torch.jit.script(MyModule())
        example_input = torch.randn(4, 3)
        m(example_input)
        torch.jit.save(m,"scriptmodule.pt")
        """
    )
    paddle_code = textwrap.dedent(
        """
        import paddle


        class MyModule(paddle.nn.Layer):
            def __init__(self):
                super(MyModule, self).__init__()
                self.dummy_param = paddle.nn.parameter.Parameter(paddle.to_tensor(data=[1.0]))

            def forward(self, x):
                return x + 10


        m = paddle.jit.to_static(function=MyModule())
        example_input = paddle.randn(shape=[4, 3])
        m(example_input)
        paddle.jit.save(layer=m, path="scriptmodule.pt".rsplit(".", 1)[0])
        """
    )
    obj.run(pytorch_code, expect_paddle_code=paddle_code)


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn

        class MyModule(nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.dummy_param = nn.Parameter(torch.tensor([1.0]))

            def forward(self, x):
                return x + 10


        m = torch.jit.script(MyModule())
        example_input = torch.randn(4, 3)
        m(example_input)
        torch.jit.save(m=m,f="scriptmodule.pt")
        """
    )
    paddle_code = textwrap.dedent(
        """
        import paddle


        class MyModule(paddle.nn.Layer):
            def __init__(self):
                super(MyModule, self).__init__()
                self.dummy_param = paddle.nn.parameter.Parameter(paddle.to_tensor(data=[1.0]))

            def forward(self, x):
                return x + 10


        m = paddle.jit.to_static(function=MyModule())
        example_input = paddle.randn(shape=[4, 3])
        m(example_input)
        paddle.jit.save(layer=m, path="scriptmodule.pt".rsplit(".", 1)[0])
        """
    )
    obj.run(pytorch_code, expect_paddle_code=paddle_code)


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn

        class MyModule(nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.dummy_param = nn.Parameter(torch.tensor([1.0]))

            def forward(self, x):
                return x + 10


        m = torch.jit.script(MyModule())
        example_input = torch.randn(4, 3)
        m(example_input)
        file_path = "scriptmodule.pt"
        torch.jit.save(m=m,f=file_path)
        """
    )
    paddle_code = textwrap.dedent(
        """
        import paddle


        class MyModule(paddle.nn.Layer):
            def __init__(self):
                super(MyModule, self).__init__()
                self.dummy_param = paddle.nn.parameter.Parameter(paddle.to_tensor(data=[1.0]))

            def forward(self, x):
                return x + 10


        m = paddle.jit.to_static(function=MyModule())
        example_input = paddle.randn(shape=[4, 3])
        m(example_input)
        file_path = "scriptmodule.pt"
        paddle.jit.save(layer=m, path=file_path.rsplit(".", 1)[0])
        """
    )
    obj.run(pytorch_code, expect_paddle_code=paddle_code)


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn

        class MyModule(nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.dummy_param = nn.Parameter(torch.tensor([1.0]))

            def forward(self, x):
                return x + 10


        m = torch.jit.script(MyModule())
        example_input = torch.randn(4, 3)
        m(example_input)
        file_path = "script.module.pt"
        torch.jit.save(m=m,f=file_path)
        """
    )
    paddle_code = textwrap.dedent(
        """
        import paddle


        class MyModule(paddle.nn.Layer):
            def __init__(self):
                super(MyModule, self).__init__()
                self.dummy_param = paddle.nn.parameter.Parameter(paddle.to_tensor(data=[1.0]))

            def forward(self, x):
                return x + 10


        m = paddle.jit.to_static(function=MyModule())
        example_input = paddle.randn(shape=[4, 3])
        m(example_input)
        file_path = "script.module.pt"
        paddle.jit.save(layer=m, path=file_path.rsplit(".", 1)[0])
        """
    )
    obj.run(pytorch_code, expect_paddle_code=paddle_code)
