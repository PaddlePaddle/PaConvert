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

obj = APIBase("torch.Tensor.fmod")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([1., 2., 3., 4., 5.])
        result = a.fmod(1.5)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.tensor([3., 2, 1, 1, 2, 3]).fmod(2.)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.tensor([3., 2, 1, 1, 2, 3]).fmod(other=torch.tensor([2.]))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.tensor([[3., 2, 1], [1, 2, 3]]).fmod(other=torch.tensor([2., 3, 1]))
        """
    )
    obj.run(pytorch_code, ["result"])


# paddle.Tensor.mod not support type promote and x/y must have same dtype
def _test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.tensor([[3., 2, 1], [1, 2, 3]]).fmod(other=torch.tensor([2, 3, 1]))
        """
    )
    obj.run(pytorch_code, ["result"])
