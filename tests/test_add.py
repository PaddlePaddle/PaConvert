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

obj = APIBase("torch.add")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.add(torch.tensor([1, 2, 3]), torch.tensor([1, 4, 6]))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.add(torch.tensor([1, 2, 3]), 20)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([20])
        result = torch.add(a, b)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([20])
        result = torch.add(a, b, alpha = 10)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([1, 2, 3])
        result = torch.add(a, torch.tensor([1, 4, 6]), alpha = 10, out=a)
        """
    )
    obj.run(pytorch_code, ["a"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([1, 2, 3])
        result = torch.add(input=a, other=torch.tensor([1, 4, 6]), alpha = 10, out=a)
        """
    )
    obj.run(pytorch_code, ["a"])


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([1, 2, 3])
        result = torch.add(other=torch.tensor([1, 4, 6]), out=a, input=a, alpha = 10)
        """
    )
    obj.run(pytorch_code, ["a"])
