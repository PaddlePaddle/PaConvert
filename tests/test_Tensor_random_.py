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

obj = APIBase("torch.Tensor.random_")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.tensor([-0.6341, -1.4208, -1.0900,  0.5826]).random_(0, 5)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([-0.6341, -1.4208, -1.0900,  0.5826])
        result = input.random_(0, 5)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([-0.6341, -1.4208, -1.0900,  0.5826])
        result = input.random_(0, to=5)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([-0.6341, -1.4208, -1.0900,  0.5826])
        result = input.random_(0, to=5, generator=None)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)
