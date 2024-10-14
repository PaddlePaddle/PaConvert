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

obj = APIBase("torch.hamming_window")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.hamming_window(10)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.hamming_window(10, dtype=torch.float64)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a=10
        result = torch.hamming_window(window_length=a)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.hamming_window(window_length=10)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.hamming_window(10, periodic=False)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.hamming_window(10, periodic=False, dtype=torch.float64, layout=torch.strided, requires_grad=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.hamming_window(10, periodic=False, dtype=torch.float64, layout=torch.strided, device=torch.device('cpu'), requires_grad=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.hamming_window(dtype=torch.float64, layout=torch.strided, window_length=5, requires_grad=True, device=torch.device('cpu'))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.hamming_window(layout=torch.strided, dtype=torch.float64, requires_grad=True, device=torch.device('cpu'), window_length=5, periodic=False)
        """
    )
    obj.run(pytorch_code, ["result"])
