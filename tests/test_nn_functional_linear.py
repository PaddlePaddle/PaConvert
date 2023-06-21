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

obj = APIBase("torch.nn.functional.linear")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.tensor([[ 0.4730,  0.5108], [ 0.4490, -0.3028], [-2.7290,  0.1999]])
        weight = torch.tensor([[ 0.5023,  1.7030], [-1.0364, -0.9937], [ 0.5375, -0.0217], [-0.2975,  0.2248]])
        bias = torch.tensor([1., 1., 1., 1.])
        result = F.linear(x, weight, bias)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.tensor([[ 0.4730,  0.5108], [ 0.4490, -0.3028], [-2.7290,  0.1999]])
        weight = torch.tensor([[ 0.5023,  1.7030], [-1.0364, -0.9937], [ 0.5375, -0.0217], [-0.2975,  0.2248]])
        result = F.linear(x, weight)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.tensor([[ 0.4730,  0.5108], [ 0.4490, -0.3028], [-2.7290,  0.1999]])
        weight = torch.tensor([[ 0.5023,  1.7030], [-1.0364, -0.9937], [ 0.5375, -0.0217], [-0.2975,  0.2248]])
        bias = torch.tensor([1., 1., 1., 1.])
        result = F.linear(input=x, weight=weight, bias=bias)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.tensor([[ 0.4730,  0.5108], [ 0.4490, -0.3028], [-2.7290,  0.1999]])
        weight = torch.tensor([[ 0.5023,  1.7030], [-1.0364, -0.9937], [ 0.5375, -0.0217], [-0.2975,  0.2248]])
        result = F.linear(x, weight=weight, bias = torch.tensor([1., 1., 1., 1.]))
        """
    )
    obj.run(pytorch_code, ["result"])
