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

import os
import sys

sys.path.append(os.path.dirname(__file__) + "/../")
import textwrap

from tests.apibase import APIBase

obj = APIBase("torch.Tensor.cholesky_inverse")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[ 0.9967,  0.0000,  0.0000],
            [-0.6374,  0.6860,  0.0000],
            [ 1.5858, -1.0314,  2.6615]])
        result = a.cholesky_inverse()
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[ 0.9967, -0.6374,  1.5858],
            [ 0.0000,  0.6860, -1.0314],
            [ 0.0000,  0.0000,  2.6615]])
        result = a.cholesky_inverse(upper=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[ 0.9967, -0.6374,  1.5858],
            [ 0.0000,  0.6860, -1.0314],
            [ 0.0000,  0.0000,  2.6615]])
        result = a.cholesky_inverse(upper=True) + 1
        """
    )
    obj.run(pytorch_code, ["result"])
