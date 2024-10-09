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
#

import textwrap

from apibase import APIBase

obj = APIBase("torch.Tensor.cauchy_")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.tensor([-0.6341, -1.4208, -1.0900,  0.5826]).cauchy_()
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([-0.6341, -1.4208, -1.0900,  0.5826])
        result = input.cauchy_()
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)

def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([-0.6341, -1.4208, -1.0900,  0.5826])
        result = input.cauchy_(median=0)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)

def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([-0.6341, -1.4208, -1.0900,  0.5826])
        result = input.cauchy_(median=0, sigma=1)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)

def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([-0.6341, -1.4208, -1.0900,  0.5826])
        result = input.cauchy_(median=0, sigma=1, generator=None)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)