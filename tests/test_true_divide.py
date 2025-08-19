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

obj = APIBase("torch.true_divide")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([4.67, 9.76 , 8.53])
        b = torch.tensor([3.5, 3.90, 1.83])
        result = torch.true_divide(a, b)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[4., 9., 8.]])
        b = torch.tensor([2., 3., 4.])
        result = torch.true_divide(a, b)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([4.67, 9.76 , 8.53])
        b = torch.tensor([3.5, 3.90, 1.83])
        out = torch.tensor([4.67, 9.76 , 8.53])
        result = torch.true_divide(a, other=b, out=out)
        """
    )
    obj.run(pytorch_code, ["result", "out"])


# def test_case_4():
#     pytorch_code = textwrap.dedent(
#         """
#         import torch
#         a = torch.tensor([4.67, 9.76 , 8.53])
#         result = torch.true_divide(a, 2.0)
#         """
#     )
#     obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[4, 9, 8]])
        b = torch.tensor([2, 3, 4])
        result = torch.true_divide(input=a, other=b)
        """
    )
    obj.run(pytorch_code, ["result"])


def _test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[4, 3, 8]])
        b = torch.tensor([3, 2, 5])
        result = torch.true_divide(input=a, other=b)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([4.67, 9.76 , 8.53])
        b = torch.tensor([3.5, 3.90, 1.83])
        out = torch.tensor([4.67, 9.76 , 8.53])
        result = torch.true_divide(input=a, other=b, out=out)
        """
    )
    obj.run(pytorch_code, ["result", "out"])


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([4.67, 9.76 , 8.53])
        b = torch.tensor([3.5, 3.90, 1.83])
        out = torch.tensor([4.67, 9.76 , 8.53])
        result = torch.true_divide(other=b, input=a, out=out)
        """
    )
    obj.run(pytorch_code, ["result", "out"])
