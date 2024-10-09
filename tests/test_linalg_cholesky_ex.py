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

obj = APIBase("torch.linalg.cholesky_ex")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[1.07676095, 1.34574506, 0.74611458],
        [1.34574506, 2.00152669, 1.24800785],
        [0.74611458, 1.24800785, 0.88039371]])
        out, info = torch.linalg.cholesky_ex(a)
        """
    )
    obj.run(pytorch_code, ["out", "info"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[1.07676095, 1.34574506, 0.74611458],
        [1.34574506, 2.00152669, 1.24800785],
        [0.74611458, 1.24800785, 0.88039371]])
        out, info = torch.linalg.cholesky_ex(a, upper=False)
        """
    )
    obj.run(pytorch_code, ["out", "info"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[1.07676095, 1.34574506, 0.74611458],
        [1.34574506, 2.00152669, 1.24800785],
        [0.74611458, 1.24800785, 0.88039371]])
        out1 = torch.randn(3, 3)
        info1 = torch.tensor([1, 2, 3], dtype=torch.int32)
        out1, info1 = torch.linalg.cholesky_ex(a, upper=True, out=(out1, info1))
        """
    )
    obj.run(pytorch_code, ["out1", "info1"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[1.07676095, 1.34574506, 0.74611458],
        [1.34574506, 2.00152669, 1.24800785],
        [0.74611458, 1.24800785, 0.88039371]])
        out1 = torch.randn(3, 3)
        info1 = torch.tensor([1, 2, 3], dtype=torch.int32)
        torch.linalg.cholesky_ex(a, check_errors=False, upper=True, out=(out1, info1))
        """
    )
    obj.run(pytorch_code, ["out1", "info1"])
