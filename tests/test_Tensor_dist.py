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

obj = APIBase("torch.Tensor.dist")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([-1.5393, -0.8675,  0.5916,  1.6321])
        other = torch.tensor([ 0.0967, -1.0511,  0.6295,  0.8360])
        result = input.dist(other, 2)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([-1.5393, -0.8675,  0.5916,  1.6321])
        other = torch.tensor([ 0.0967, -1.0511,  0.6295,  0.8360])
        result = input.dist(other, p=2.5)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([-1.5393, -0.8675,  0.5916,  1.6321])
        other = torch.tensor([ 0.0967, -1.0511,  0.6295,  0.8360])
        p = 3
        result = input.dist(other, p=p)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([-1.5393, -0.8675,  0.5916,  1.6321])
        other = torch.tensor([ 0.0967])
        result = input.dist(other, 2)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.tensor([-1.5393, -0.8675, 0.5916, 1.6321]).dist(other=torch.tensor([ 0.0967]), p=2)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.tensor([-1.5393, -0.8675, 0.5916, 1.6321]).dist(p=2, other=torch.tensor([ 0.0967]))
        """
    )
    obj.run(pytorch_code, ["result"])
