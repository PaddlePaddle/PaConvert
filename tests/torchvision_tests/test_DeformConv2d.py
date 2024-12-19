# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

obj = APIBase("torchvision.ops.DeformConv2d")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torchvision
        deform_conv = torchvision.ops.DeformConv2d(
            in_channels=3,
            out_channels=4,
            kernel_size=3,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
        )
        """
    )
    paddle_code = textwrap.dedent(
        """
        import paddle

        deform_conv = paddle.vision.ops.DeformConv2D(
            in_channels=3,
            out_channels=4,
            kernel_size=3,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
        )
        """
    )
    obj.run(pytorch_code, expect_paddle_code=paddle_code, check_value=False)


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torchvision
        deform_conv = torchvision.ops.DeformConv2d(3, 4, 3, 1, 0, 1, 1)
        """
    )
    paddle_code = textwrap.dedent(
        """
        import paddle

        deform_conv = paddle.vision.ops.DeformConv2D(
            in_channels=3,
            out_channels=4,
            kernel_size=3,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
        )
        """
    )
    obj.run(pytorch_code, expect_paddle_code=paddle_code, check_value=False)


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torchvision
        deform_conv = torchvision.ops.DeformConv2d(
            kernel_size=3,
            in_channels=3,
            out_channels=64,
            padding=1,
            stride=1,
            dilation=1,
            groups=1,
        )
        """
    )
    paddle_code = textwrap.dedent(
        """
        import paddle

        deform_conv = paddle.vision.ops.DeformConv2D(
            kernel_size=3,
            in_channels=3,
            out_channels=64,
            padding=1,
            stride=1,
            dilation=1,
            groups=1,
        )
        """
    )
    obj.run(pytorch_code, expect_paddle_code=paddle_code, check_value=False)


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torchvision
        deform_conv = torchvision.ops.DeformConv2d(3, 64, 3)
        """
    )
    paddle_code = textwrap.dedent(
        """
        import paddle

        deform_conv = paddle.vision.ops.DeformConv2D(
            in_channels=3, out_channels=64, kernel_size=3
        )
        """
    )
    obj.run(pytorch_code, expect_paddle_code=paddle_code, check_value=False)


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torchvision
        deform_conv = torchvision.ops.DeformConv2d(
            groups=1,
            dilation=1,
            padding=1,
            stride=1,
            kernel_size=3,
            out_channels=64,
            in_channels=3
        )
        """
    )
    paddle_code = textwrap.dedent(
        """
        import paddle

        deform_conv = paddle.vision.ops.DeformConv2D(
            groups=1,
            dilation=1,
            padding=1,
            stride=1,
            kernel_size=3,
            out_channels=64,
            in_channels=3,
        )
        """
    )
    obj.run(pytorch_code, expect_paddle_code=paddle_code, check_value=False)
