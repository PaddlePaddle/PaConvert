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

obj = APIBase("torch.max_pool1d")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[[ 1.1524,  0.4714,  0.2857],
            [-1.2533, -0.9829, -1.0981],
            [ 0.1507, -1.1431, -2.0361]]])
        result = torch.max_pool1d(input , 3)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[[ 1.1524,  0.4714,  0.2857],
            [-1.2533, -0.9829, -1.0981],
            [ 0.1507, -1.1431, -2.0361]]])
        result = torch.max_pool1d(input , 3, stride=2)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[[ 1.1524,  0.4714,  0.2857, 0.4586, 0.9876, 0.5487],
            [-1.2533, -0.9829, -1.0981, 0.7655, 0.8541, 0.9873],
            [ 0.1507, -1.1431, -2.0361, 0.2344, 0.5675, 0.1546]]])
        result = torch.max_pool1d(input , 5, stride=2, padding=2)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[[ 1.1524,  0.4714,  0.2857, 0.4586, 0.9876, 0.5487],
            [-1.2533, -0.9829, -1.0981, 0.7655, 0.8541, 0.9873],
            [ 0.1507, -1.1431, -2.0361, 0.2344, 0.5675, 0.1546]]])
        result = torch.max_pool1d(input, 5, stride=2, padding=2, ceil_mode=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[[ 1.1524,  0.4714,  0.2857, 0.4586, 0.9876, 0.5487],
            [-1.2533, -0.9829, -1.0981, 0.7655, 0.8541, 0.9873],
            [ 0.1507, -1.1431, -2.0361, 0.2344, 0.5675, 0.1546]]])
        result = torch.max_pool1d(input=input, kernel_size=5, stride=2, padding=2, ceil_mode=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[[ 1.1524,  0.4714,  0.2857, 0.4586, 0.9876, 0.5487],
            [-1.2533, -0.9829, -1.0981, 0.7655, 0.8541, 0.9873],
            [ 0.1507, -1.1431, -2.0361, 0.2344, 0.5675, 0.1546]]])
        result = torch.max_pool1d(input=input, padding=2, kernel_size=5, stride=2, ceil_mode=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[[ 1.1524,  0.4714,  0.2857, 0.4586, 0.9876, 0.5487],
            [-1.2533, -0.9829, -1.0981, 0.7655, 0.8541, 0.9873],
            [ 0.1507, -1.1431, -2.0361, 0.2344, 0.5675, 0.1546]]])
        result = torch.max_pool1d(input=input, kernel_size=5, stride=2, padding=2, dilation=1, ceil_mode=True)
        """
    )
    obj.run(pytorch_code, unsupport=True, reason="Not support dilation")


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[[ 1.1524,  0.4714,  0.2857, 0.4586, 0.9876, 0.5487],
            [-1.2533, -0.9829, -1.0981, 0.7655, 0.8541, 0.9873],
            [ 0.1507, -1.1431, -2.0361, 0.2344, 0.5675, 0.1546]]])
        result = torch.max_pool1d(input, 5, 2, 2, 1, True)
        """
    )
    obj.run(pytorch_code, unsupport=True, reason="Not support dilation")
