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

obj = APIBase("torch.pow")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([0.4331,  1.2475,  0.6834, -0.2791])
        result = torch.pow(a, 2)
        """
    )
    obj.run(pytorch_code, ["result"])


# The data types of two inputs must be the same. When two input types are different, type conversion is not performed automatically.
def _test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([0.4331,  1.2475,  0.6834, -0.2791])
        b = torch.tensor([1, 2, 3, 4])
        result = torch.pow(a, b)
        """
    )
    obj.run(pytorch_code, ["result"])


# The data types of two inputs must be the same. When two input types are different, type conversion is not performed automatically.
def _test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([0.4331])
        b = torch.tensor([1, 2, 3, 4])
        result = torch.pow(a, b)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([2])
        b = torch.tensor([1, 2, 3, 4])
        result = torch.pow(a, b)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([1, 2, 3, 4])
        out = torch.tensor([1, 2, 3, 4])
        result = torch.pow(a, 2, out=out)
        """
    )
    obj.run(pytorch_code, ["result", "out"])
