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

import textwrap

from apibase import APIBase

obj = APIBase("torch.frombuffer")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import array
        a = array.array('i', [1, 2, 3])
        result = torch.frombuffer(a, dtype=torch.int32)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import array
        a = array.array('b', [-1, 0, 0, 0])
        result = torch.frombuffer(a, dtype=torch.int8)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import array
        a = array.array('h', [1, 2])
        result = torch.frombuffer(a, dtype=torch.int16)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import array
        a = array.array('i', [1, 2, 3, 4])
        result = torch.frombuffer(a, dtype=torch.int64)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import array
        a = array.array('d', [0.1, 0.4, 0.3, 0.2, 0.5, 0.8, 0.6, 0.9])
        result = torch.frombuffer(buffer=a, dtype=torch.int64)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import array
        a = array.array('l', [2, 1, 4, 3])
        result = torch.frombuffer(a, dtype=torch.float16)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import array
        result = torch.frombuffer(array.array('i', [1, 2, 3]), dtype=torch.int32)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import array
        result = torch.frombuffer(buffer=array.array('h', [1, 2]), dtype=torch.float32)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import array
        result = torch.frombuffer(buffer=array.array('h', [1, 2]), dtype=torch.uint8)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_10():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import array
        result = torch.frombuffer(dtype=torch.float64, buffer=array.array('i', [1, 2, 3, 4]))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_11():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import array
        result = torch.frombuffer(dtype=torch.complex64, buffer=array.array('i', [1, 2, 3, 4]))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_12():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import array
        result = torch.frombuffer(buffer=array.array('b', [-1, 0, 0, 0]), dtype=torch.bool)
        """
    )
    obj.run(pytorch_code, ["result"])
