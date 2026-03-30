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

obj = APIBase("torch.iinfo")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        bits = torch.iinfo(torch.int32).bits
        min = torch.iinfo(torch.int32).min
        max = torch.iinfo(torch.int32).max
        """
    )
    obj.run(pytorch_code, ["bits", "min", "max"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.int16
        bits = torch.iinfo(x).bits
        min = torch.iinfo(x).min
        max = torch.iinfo(x).max
        """
    )
    obj.run(pytorch_code, ["bits", "min", "max"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.uint8
        bits = torch.iinfo(type=x).bits
        min = torch.iinfo(type=x).min
        max = torch.iinfo(type=x).max
        """
    )
    obj.run(pytorch_code, ["bits", "min", "max"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1])
        bits = torch.iinfo(x.dtype).bits
        min = torch.iinfo(x.dtype).min
        max = torch.iinfo(x.dtype).max
        """
    )
    obj.run(pytorch_code, ["bits", "min", "max"])


def test_case_5():
    """Test with torch.int8"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        bits = torch.iinfo(torch.int8).bits
        min = torch.iinfo(torch.int8).min
        max = torch.iinfo(torch.int8).max
        """
    )
    obj.run(pytorch_code, ["bits", "min", "max"])


def test_case_6():
    """Test with torch.int64"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        bits = torch.iinfo(torch.int64).bits
        min = torch.iinfo(torch.int64).min
        max = torch.iinfo(torch.int64).max
        """
    )
    obj.run(pytorch_code, ["bits", "min", "max"])


def test_case_7():
    """Test with torch.short (int16)"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        bits = torch.iinfo(torch.short).bits
        min = torch.iinfo(torch.short).min
        max = torch.iinfo(torch.short).max
        """
    )
    obj.run(pytorch_code, ["bits", "min", "max"])


def test_case_8():
    """Test with torch.long (int64)"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        bits = torch.iinfo(torch.long).bits
        min = torch.iinfo(torch.long).min
        max = torch.iinfo(torch.long).max
        """
    )
    obj.run(pytorch_code, ["bits", "min", "max"])


# Paddle does not support uint16 type conversion
def _test_case_9():
    """Test with torch.uint16"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        bits = torch.iinfo(torch.uint16).bits
        min = torch.iinfo(torch.uint16).min
        max = torch.iinfo(torch.uint16).max
        """
    )
    obj.run(pytorch_code, ["bits", "min", "max"])


# Paddle does not support uint32 type conversion
def _test_case_10():
    """Test with torch.uint32"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        bits = torch.iinfo(torch.uint32).bits
        min = torch.iinfo(torch.uint32).min
        max = torch.iinfo(torch.uint32).max
        """
    )
    obj.run(pytorch_code, ["bits", "min", "max"])


# Paddle does not support uint64 type conversion
def _test_case_11():
    """Test with torch.uint64"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        bits = torch.iinfo(torch.uint64).bits
        min = torch.iinfo(torch.uint64).min
        max = torch.iinfo(torch.uint64).max
        """
    )
    obj.run(pytorch_code, ["bits", "min", "max"])


def test_case_12():
    """Test with variable dtype"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        dtype = torch.int32
        info = torch.iinfo(type=dtype)
        bits = info.bits
        min = info.min
        max = info.max
        """
    )
    obj.run(pytorch_code, ["bits", "min", "max"])


def test_case_13():
    """Test with int tensor dtype"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1, 2, 3], dtype=torch.int64)
        info = torch.iinfo(type=x.dtype)
        bits = info.bits
        min = info.min
        max = info.max
        """
    )
    obj.run(pytorch_code, ["bits", "min", "max"])
