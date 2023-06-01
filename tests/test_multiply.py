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

obj = APIBase("torch.multiply")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([0.2015, -0.4255, 2.6087])
        other = torch.tensor([0.2015, -0.4255, 2.6087])
        result = torch.multiply(input, other)
        """
    )
    obj.run(pytorch_code, ["result"])


# The data types of two inputs must be the same. When two input types are different, type conversion is not performed automatically.
def _test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([0.2015, -0.4255,  2.6087])
        other = torch.tensor([2, 6, 4])
        result = torch.multiply(input, other)
        """
    )
    obj.run(pytorch_code, ["result"])


# The second argument of a torch can be a Number, paddle must be tensor.
def _test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([0.2015, -0.4255,  2.6087])
        result = torch.multiply(input, other=5)
        """
    )
    obj.run(pytorch_code, ["result"])


def _test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([3, 6, 9])
        result = torch.multiply(input, other = torch.tensor([0.2015, -0.4255, 2.6087]))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([0.2015, -0.4255,  2.6087])
        out = torch.tensor([0.2015, -0.4255,  2.6087])
        result = torch.multiply(input, torch.tensor([0.2015, -0.4255, 2.6087]), out=out)
        """
    )
    obj.run(pytorch_code, ["result", "out"])
