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

obj = APIBase("torch.Tensor.mul", is_aux_api=True)


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([0.2015, -0.4255, 2.6087])
        other = torch.tensor([0.2015, -0.4255, 2.6087])
        result = input.mul(other)
        """
    )
    obj.run(pytorch_code, ["result"])


# paddle not support input type promote, and x/y must have the same dtype
def _test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([0.2015, -0.4255,  2.6087])
        other = torch.tensor([2, 6, 4])
        result = input.mul(other=other)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([0.2015, -0.4255,  2.6087])
        result = input.mul(other=torch.tensor(5.))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([3, 6, 9])
        result = input.mul(5)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([3, 6, 9])
        result = input.mul(other=5)
        """
    )
    obj.run(pytorch_code, ["result"])


# paddle not support type promote and x/y must have same dtype
def _test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([0.2015, -0.4255,  2.6087])
        other = torch.tensor([2, 6, 4])
        result = input.mul(other)
        """
    )
    obj.run(pytorch_code, ["result"])
