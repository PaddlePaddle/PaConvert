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

obj = APIBase("torch.Tensor.data")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1.3192, 1.9915, 1.9674, 1.7151])
        result = x.data
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1.3192, 1.9915, 1.9674, 1.7151])
        x.data = torch.tensor([1., 1., 1., 1.])
        result = x.data
        """
    )
    obj.run(pytorch_code, ["x", "result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        linear = torch.nn.Linear(5, 5)
        torch.nn.init.constant_(linear.weight, 1)

        result = linear.weight.data
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        linear = torch.nn.Linear(5, 5)
        linear.weight.data = torch.ones(5, 5)

        result = linear.weight.data
        """
    )
    obj.run(pytorch_code, ["result"])
