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

import os
import sys

sys.path.append(os.path.dirname(__file__) + "/../")
import textwrap

from tests.apibase import APIBase

obj = APIBase("torch.unflatten")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[ 0.2180,  1.0558,  0.1608,  0.9245],
                [ 1.3794,  1.4090,  0.2514, -0.8818],
                [-0.4561,  0.5123,  1.7505, -0.4094]])
        result = torch.unflatten(a, -1, (2, 2))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[ 0.2180,  1.0558,  0.1608,  0.9245],
                [ 1.3794,  1.4090,  0.2514, -0.8818],
                [-0.4561,  0.5123,  1.7505, -0.4094]])
        result = torch.unflatten(a, 1, (2, 2))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[ 0.2180,  1.0558,  0.1608,  0.9245],
                [ 1.3794,  1.4090,  0.2514, -0.8818],
                [-0.4561,  0.5123,  1.7505, -0.4094]])
        result = torch.unflatten(a, -1, (2, 1, 2))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[ 0.2180,  1.0558,  0.1608,  0.9245],
                [ 1.3794,  1.4090,  0.2514, -0.8818],
                [-0.4561,  0.5123,  1.7505, -0.4094]])
        result = torch.unflatten(input=a, dim=-1, sizes=(2, 1, 2))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[ 0.2180,  1.0558,  0.1608,  0.9245],
                [ 1.3794,  1.4090,  0.2514, -0.8818],
                [-0.4561,  0.5123,  1.7505, -0.4094]])
        result = torch.unflatten(input=a, dim=-1, sizes=[2, 1, 2])
        """
    )
    obj.run(pytorch_code, ["result"])
