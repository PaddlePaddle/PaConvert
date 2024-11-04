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

obj = APIBase("torch.mvlgamma")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([2.5, 3.5, 4, 6.5, 7.8, 10.23, 34.25])
        result = torch.mvlgamma(x, 2)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([2.5, 3.5, 4, 6.5, 7.8, 10.23, 34.25])
        result = torch.mvlgamma(x, p=2)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([2.5, 3.5, 4, 6.5, 7.8, 10.23, 34.25])
        result = torch.tensor([2.5, 3.5, 4, 6.5, 7.8, 10.23, 34.25])
        torch.mvlgamma(x, 2, out=result)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([2.5, 3.5, 4, 6.5, 7.8, 10.23, 34.25])
        result = torch.tensor([2.5, 3.5, 4, 6.5, 7.8, 10.23, 34.25])
        torch.mvlgamma(input=x, p=2, out=result)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([2.5, 3.5, 4, 6.5, 7.8, 10.23, 34.25])
        result = torch.tensor([2.5, 3.5, 4, 6.5, 7.8, 10.23, 34.25])
        torch.mvlgamma(out=result, input=x, p=2)
        """
    )
    obj.run(pytorch_code, ["result"])
