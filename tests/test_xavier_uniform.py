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

obj = APIBase("torch.nn.init.xavier_uniform")


def test_case_1():
    """Basic usage with positional args"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.empty(3, 5)
        torch.nn.init.xavier_uniform(x)
        result = x
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_2():
    """With gain parameter"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.empty(3, 5)
        torch.nn.init.xavier_uniform(x, gain=2.0)
        result = x
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_3():
    """With all keyword arguments"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.empty(3, 5)
        torch.nn.init.xavier_uniform(tensor=x, gain=1.5)
        result = x
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_4():
    """2D tensor initialization"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        weight = torch.empty(10, 20)
        torch.nn.init.xavier_uniform(weight)
        result = weight
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_5():
    """4D conv weight tensor initialization"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        weight = torch.empty(8, 4, 3, 3)
        torch.nn.init.xavier_uniform(weight)
        result = weight
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)
