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

obj = APIBase("torch.remainder")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([-3., -2, -1, 1, 2, 3])
        result = torch.remainder(a, 2.)
        """
    )
    obj.run(pytorch_code, ["result"])


# The paddle input does not support integer type
def _test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([1, 2, 3, 4, 5])
        result = torch.remainder(a, torch.tensor(1.5))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.remainder(torch.tensor([-3., -2, -1, 1, 2, 3]), 2.)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([1., 2, 3, 4, 5])
        b = torch.tensor([1, 0.5, 0.6, 1.2, 2.4])
        result = torch.remainder(a, b)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([1., 2, 3, 4, 5])
        out = torch.tensor([1, 0.5, 0.6, 1.2, 2.4])
        result = torch.remainder(a, 2., out=out)
        """
    )
    obj.run(pytorch_code, ["result", "out"])


# The two input types of paddle must be consistent and cannot be transformed automatically
def _test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([1, 2, 3, 4, 5])
        b = torch.tensor([1, 0.5, 0.6, 1.2, 2.4])
        result = torch.remainder(a, b)
        """
    )
    obj.run(pytorch_code, ["result"])
