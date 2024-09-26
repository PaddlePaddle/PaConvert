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

obj = APIBase("torch.Tensor.copysign")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([-1.2557, -0.0026, -0.5387,  0.4740, -0.9244])
        result = a.copysign(1.)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[ 0.7079,  0.2778, -1.0249,  0.5719],
        [-0.0059, -0.2600, -0.4475, -1.3948],
        [ 0.3667, -0.9567, -2.5757, -0.1751],
        [ 0.2046, -0.0742,  0.2998, -0.1054]])
        b = torch.tensor([ 0.2373,  0.3120,  0.3190, -1.1128])
        result = a.copysign(b)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([1.])
        b = torch.tensor([-0.])
        result = a.copysign(b)
        """
    )
    obj.run(pytorch_code, ["result"])


# paddle.Tensor.copysign not support type promote and x/y must have same dtype
def _test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.tensor([3., 2, 1]).copysign(other=torch.tensor([2]))
        """
    )
    obj.run(pytorch_code, ["result"])
