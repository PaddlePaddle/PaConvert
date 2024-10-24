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

obj = APIBase("torch.copysign")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([1, 2, 3])
        result = torch.copysign(a, -1, out=None)
        """
    )
    obj.run(pytorch_code, ["result"], check_dtype=False)


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([1, 2, 3, 4])
        b = torch.tensor([-1, 2, -3, 4])
        result = torch.copysign(a, b, out=None)
        """
    )
    obj.run(pytorch_code, ["result"], check_dtype=False)


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[0.7079, 0.2778, -1.0249, 0.5719],
                                   [-0.0059, -0.2600, -0.4475, -1.3948],
                                   [0.3667, -0.9567, -2.5757, -0.1751],
                                   [0.2046, -0.0742, 0.2998, -0.1054]])
        b = torch.tensor([-1, 2, -3, 4])
        result = torch.copysign(a, b, out=a)
        """
    )
    obj.run(pytorch_code, ["result", "a"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[0.7079, 0.2778, -1.0249, 0.5719],
                                   [-0.0059, -0.2600, -0.4475, -1.3948],
                                   [0.3667, -0.9567, -2.5757, -0.1751],
                                   [0.2046, -0.0742, 0.2998, -0.1054]])
        b = torch.tensor([-1, 2, -3, 4])
        result = torch.copysign(input=a, other=b, out=a)
        """
    )
    obj.run(pytorch_code, ["result", "a"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[0.7079, 0.2778, -1.0249, 0.5719],
                                   [-0.0059, -0.2600, -0.4475, -1.3948],
                                   [0.3667, -0.9567, -2.5757, -0.1751],
                                   [0.2046, -0.0742, 0.2998, -0.1054]])
        result = torch.copysign(a, -1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[0.7079, 0.2778, -1.0249, 0.5719],
                                   [-0.0059, -0.2600, -0.4475, -1.3948],
                                   [0.3667, -0.9567, -2.5757, -0.1751],
                                   [0.2046, -0.0742, 0.2998, -0.1054]])
        b = torch.tensor([-1, 2, -3, 4])
        result = torch.copysign(out=a, input=a, other=b)
        """
    )
    obj.run(pytorch_code, ["result", "a"])
