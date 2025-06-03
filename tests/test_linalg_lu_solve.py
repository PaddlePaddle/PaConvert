# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

obj = APIBase("torch.linalg.lu_solve")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        A = torch.tensor([[4.0, 3.0], [6.0, 3.0]])
        B_left = torch.tensor([[1.0], [2.0]])
        LU, pivots = torch.linalg.lu_factor(A)
        result = torch.linalg.lu_solve(LU, pivots, B_left)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        A = torch.tensor([[4.0, 3.0], [6.0, 3.0]])
        B_left = torch.tensor([[1.0], [2.0]])
        LU, pivots = torch.linalg.lu_factor(A)
        output = torch.empty_like(B_left)
        result = torch.linalg.lu_solve(LU=LU, pivots=pivots, B=B_left, left=True, adjoint=True, out=output)
        """
    )
    obj.run(pytorch_code, ["result", "output"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        A = torch.tensor([[4.0, 3.0], [6.0, 3.0]])
        B_right = torch.tensor([[1.0, 2.0]])
        LU, pivots = torch.linalg.lu_factor(A)
        output = torch.empty_like(B_right)
        result = torch.linalg.lu_solve(LU=LU, pivots=pivots, B=B_right, out=output, left=False, adjoint=True)
        """
    )
    obj.run(pytorch_code, ["result", "output"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        A = torch.tensor([[4.0, 3.0], [6.0, 3.0]])
        B_right = torch.tensor([[1.0, 2.0]])
        LU, pivots = torch.linalg.lu_factor(A)
        output = torch.empty_like(B_right)
        result = torch.linalg.lu_solve(LU=LU, pivots=pivots, B=B_right, out=output, left=False, adjoint=False)
        """
    )
    obj.run(pytorch_code, ["result", "output"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        A = torch.tensor([[4.0, 3.0], [6.0, 3.0]])
        B_right = torch.tensor([[1.0, 2.0]])
        LU, pivots = torch.linalg.lu_factor(A)
        left = False
        result = torch.linalg.lu_solve(LU=LU, pivots=pivots, B=B_right, left=left, adjoint=False)
        """
    )
    obj.run(pytorch_code, ["result"])
