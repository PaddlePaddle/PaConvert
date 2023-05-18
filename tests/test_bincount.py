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

obj = APIBase("torch.bincount")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([4, 3, 6, 3, 4])
        weights = torch.tensor([ 0.0000,  0.2500,  0.5000,  0.7500,  1.0000])
        result = torch.bincount(input)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([4, 3, 6, 3, 4])
        weights = torch.tensor([ 0.0000,  0.2500,  0.5000,  0.7500,  1.0000])
        result = torch.bincount(input, weights)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([4, 3, 6, 3, 4])
        out = torch.tensor([4, 3, 6, 3, 4])
        weights = torch.tensor([ 0.0000,  0.2500,  0.5000,  0.7500,  1.0000])
        result = torch.bincount(input, minlength = 6)
        """
    )
    obj.run(pytorch_code, ["result", "out"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([4, 3, 6, 3, 4])
        weights = torch.tensor([ 0.0000,  0.2500,  0.5000,  0.7500,  1.0000])
        result = torch.bincount(input=input, weights=weights)
        """
    )
    obj.run(pytorch_code, ["result"])
