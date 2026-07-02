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
    obj.run(pytorch_code, ["result"])


def test_case_2():
    """Integer dtype attribute access."""
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.int64.itemsize
        """
    )
    obj.run(pytorch_code, ["result"])


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
    obj.run(pytorch_code, ["a", "b", "c", "d"])


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
    obj.run(pytorch_code, ["a", "b", "c"])


def test_case_5():
    """Remaining mapped dtypes: float16/float64/complex64/int8/int16/int32/uint8."""
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.float16.itemsize
        b = torch.float64.itemsize
        c = torch.complex64.itemsize
        d = torch.int8.itemsize
        e = torch.int16.itemsize
        f = torch.int32.itemsize
        g = torch.uint8.itemsize
        """
    )
    obj.run(pytorch_code, ["a", "b", "c", "d", "e", "f", "g"])
