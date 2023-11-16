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

obj = APIBase("torch.Tensor.triangular_solve")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[ 1.1527, -1.0753], [ 1.23,  0.7986]])
        b = torch.tensor([[-0.0210,  2.3513, -1.5492], [ 1.5429,  0.7403, -1.0243]])
        result1, result2 = b.triangular_solve(a)
        """
    )
    obj.run(pytorch_code, ["result1", "result2"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[ 1.1527, -1.0753], [ 0.0000,  0.7986]])
        b = torch.tensor([[-0.0210,  2.3513, -1.5492], [ 1.5429,  0.7403, -1.0243]])
        result1, result2 = b.triangular_solve(a, False)
        """
    )
    obj.run(pytorch_code, ["result1", "result2"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[ 1.1527, -1.0753], [ 0.0000,  0.7986]])
        b = torch.tensor([[-0.0210,  2.3513, -1.5492], [ 1.5429,  0.7403, -1.0243]])
        result1, result2 = b.triangular_solve(transpose=False, A=a)
        """
    )
    obj.run(pytorch_code, ["result1", "result2"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[ 1.1527, -1.0753], [ 0.0000,  0.7986]])
        b = torch.tensor([[-0.0210,  2.3513, -1.5492], [ 1.5429,  0.7403, -1.0243]])
        result1, result2 = b.triangular_solve(A=a, upper=True, transpose=False, unitriangular=False)
        """
    )
    obj.run(pytorch_code, ["result1", "result2"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[ 1.1527, -1.0753], [ 0.0000,  0.7986]])
        b = torch.tensor([[-0.0210,  2.3513, -1.5492], [ 1.5429,  0.7403, -1.0243]])
        result1, result2 = b.triangular_solve(a, True, False, False)
        """
    )
    obj.run(pytorch_code, ["result1", "result2"])
