# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

obj = APIBase("torch.dtype.itemsize")


def test_case_1():
    """Float dtype attribute access."""
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.float32.itemsize
        """
    )
    expect = textwrap.dedent(
        """
        import paddle

        result = paddle.float32.itemsize
        """
    )
    obj.run(pytorch_code, expect_paddle_code=expect)


def test_case_2():
    """Integer dtype attribute access."""
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.int64.itemsize
        """
    )
    expect = textwrap.dedent(
        """
        import paddle

        result = paddle.int64.itemsize
        """
    )
    obj.run(pytorch_code, expect_paddle_code=expect)


def test_case_3():
    """Complex / bool / bfloat16 / float8 — full coverage of less-common dtypes."""
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.complex128.itemsize
        b = torch.bool.itemsize
        c = torch.bfloat16.itemsize
        d = torch.float8_e4m3fn.itemsize
        """
    )
    expect = textwrap.dedent(
        """
        import paddle

        a = paddle.complex128.itemsize
        b = paddle.bool.itemsize
        c = paddle.bfloat16.itemsize
        d = paddle.float8_e4m3fn.itemsize
        """
    )
    obj.run(pytorch_code, expect_paddle_code=expect)


def test_case_4():
    """Aliases: long, double, cfloat, etc."""
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.long.itemsize
        b = torch.double.itemsize
        c = torch.cfloat.itemsize
        """
    )
    expect = textwrap.dedent(
        """
        import paddle

        a = paddle.long.itemsize
        b = paddle.double.itemsize
        c = paddle.cfloat.itemsize
        """
    )
    obj.run(pytorch_code, expect_paddle_code=expect)
