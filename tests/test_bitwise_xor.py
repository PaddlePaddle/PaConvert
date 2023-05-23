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

obj = APIBase("torch.bitwise_xor")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.bitwise_xor(torch.tensor([-1, -2, 3], dtype=torch.int8), torch.tensor([1, 0, 3], dtype=torch.int8))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([-1, -2, 3], dtype=torch.int8)
        other = torch.tensor([1, 0, 3], dtype=torch.int8)
        result = torch.bitwise_xor(input, other)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([True, False, True])
        other = torch.tensor([False, True, True])
        result = torch.bitwise_xor(input, other)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.bitwise_xor(torch.tensor([True, False, True]), torch.tensor([False, True, True]))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([True, False, True])
        other = torch.tensor([False, True, True])
        out = torch.tensor([True, False, False])
        result = torch.bitwise_xor(input, other, out=out)
        """
    )
    obj.run(pytorch_code, ["result", "out"])
