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

import pytest
from apibase import APIBase

obj = APIBase("torch.jit.load")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.jit.load('model.pt')
        """
    )
    paddle_code = textwrap.dedent(
        """
        import paddle

        result = paddle.jit.load(path="model.pt")
        """
    )
    obj.run(pytorch_code, expect_paddle_code=paddle_code)


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.jit.load(f='model.pt')
        """
    )
    paddle_code = textwrap.dedent(
        """
        import paddle

        result = paddle.jit.load(path="model.pt")
        """
    )
    obj.run(pytorch_code, expect_paddle_code=paddle_code)


@pytest.mark.skip(
    reason="torch.jit.load with map_location parameter is not supported in PaConvert currently"
)
def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.jit.load(f='model.pt', map_location='cpu')
        """
    )
    paddle_code = textwrap.dedent(
        """
        import paddle

        result = paddle.jit.load(path="model.pt")
        """
    )
    obj.run(pytorch_code, expect_paddle_code=paddle_code)
