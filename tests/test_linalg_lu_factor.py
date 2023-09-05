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

obj = APIBase("torch.linalg.lu_factor")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float64)
        LU, pivots = torch.linalg.lu_factor(x)
        """
    )
    obj.run(pytorch_code, ["LU", "pivots"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float64)
        LU, pivots = torch.linalg.lu_factor(A=x)
        """
    )
    obj.run(pytorch_code, ["LU", "pivots"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float64)
        LU, pivots = torch.linalg.lu_factor(pivot=True, A=x)
        """
    )
    obj.run(pytorch_code, ["LU", "pivots"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float64)
        out = (torch.tensor([], dtype=torch.float64), torch.tensor([], dtype=torch.int))
        LU, pivots = torch.linalg.lu_factor(x, pivot=True, out=out)
        """
    )
    obj.run(pytorch_code, ["LU", "pivots", "out"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float64)
        out = (torch.tensor([], dtype=torch.float64), torch.tensor([], dtype=torch.int))
        LU, pivots = torch.linalg.lu_factor(A=x, pivot=True, out=out)
        """
    )
    obj.run(pytorch_code, ["LU", "pivots", "out"])
