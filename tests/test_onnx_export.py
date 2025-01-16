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

obj = APIBase("torch.onnx.export")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.onnx
        import torch.nn as nn

        class SimpleModel(nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.fc1 = nn.Linear(3, 3)
                self.fc2 = nn.Linear(3, 1)

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = self.fc2(x)
                return x

        model = SimpleModel()
        x = torch.randn(1, 3)
        a = torch.onnx.export(
            model,
            f="simple_model.onnx",
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        unsupport=True,
        reason="paddle does not support args temporarily",
    )


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.onnx
        import torch.nn as nn
        class SimpleModel(nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.fc1 = nn.Linear(3, 3)
                self.fc2 = nn.Linear(3, 1)

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = self.fc2(x)
                return x

        model = SimpleModel()
        x = torch.randn(1, 3)
        a = torch.onnx.export(
            model,
            f="simple_model.onnx",
        )
        """
    )
    paddle_code = textwrap.dedent(
        """
        import paddle

        ############################## 相关utils函数，如下 ##############################

        def onnx_export(model,f):
            model = Logic()
            paddle.jit.to_static(model)
            last_dot_index = filename.rfind('.')
            if last_dot_index == -1:
                path = f
            else:
                path = f[:last_dot_index]
            return paddle.onnx.export(model, path)
        ############################## 相关utils函数，如上 ##############################



        class SimpleModel(paddle.nn.Layer):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.fc1 = paddle.nn.Linear(in_features=3, out_features=3)
                self.fc2 = paddle.nn.Linear(in_features=3, out_features=1)

            def forward(self, x):
                x = paddle.nn.functional.relu(x=self.fc1(x))
                x = self.fc2(x)
                return x


        model = SimpleModel()
        x = paddle.randn(shape=[1, 3])
        a = onnx_export(model, "simple_model.onnx")
        """
    )
    obj.run(pytorch_code, expect_paddle_code=paddle_code)
