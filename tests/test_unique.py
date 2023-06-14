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

import textwrap

from apibase import APIBase

obj = APIBase("torch.unique")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor(
            [[2, 1, 3], [3, 0, 1], [2, 1, 3]])
        result = torch.unique(a)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor(
            [[2, 1, 3], [3, 0, 1], [2, 1, 3]])
        result = torch.unique(input=a, return_inverse=True, return_counts=True, dim=1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor(
            [[2, 1, 3], [3, 0, 1], [2, 1, 3]])
        dim = 1
        result = torch.unique(input=a, return_inverse=True, return_counts=False, dim=dim)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor(
            [[2, 1, 3], [3, 0, 1], [2, 1, 3]])
        dim = 1
        result = torch.unique(input=a, sorted=False, return_inverse=True, return_counts=False, dim=dim)
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        unsupport=True,
        reason="Paddle does not currently support the 'sorted' input parameter.",
    )
