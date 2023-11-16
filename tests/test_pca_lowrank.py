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

obj = APIBase("torch.pca_lowrank")
ATOL = 1e-7


# Notice: In paddle, the cpu version and the gpu version symbols are different.
def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, -2], [2, 5]])
        u, s, v = torch.pca_lowrank(x)
        u = torch.abs(u)
        v = torch.abs(v)
        """
    )
    obj.run(pytorch_code, ["u", "s", "v"], atol=ATOL)


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, -2], [2, 5]])
        u, s, v = torch.pca_lowrank(A=x)
        u = torch.abs(u)
        v = torch.abs(v)
        """
    )
    obj.run(pytorch_code, ["u", "s", "v"], atol=ATOL)


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, -2], [2, 5]])
        out = torch.tensor([])
        u, s, v = torch.pca_lowrank(niter=2, A=x)
        u = torch.abs(u)
        v = torch.abs(v)
        """
    )
    obj.run(pytorch_code, ["u", "s", "v"], atol=ATOL)


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, -2], [2, 5]])
        out = torch.tensor([])
        u, s, v = torch.pca_lowrank(A=x, q=None, center=True, niter=2)
        u = torch.abs(u)
        v = torch.abs(v)
        """
    )
    obj.run(pytorch_code, ["u", "s", "v"], atol=ATOL)


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, -2], [2, 5]])
        u, s, v = torch.pca_lowrank(x, None, True, 2)
        u = torch.abs(u)
        v = torch.abs(v)
        """
    )
    obj.run(pytorch_code, ["u", "s", "v"], atol=ATOL)
