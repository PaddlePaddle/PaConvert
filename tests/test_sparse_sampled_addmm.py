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

obj = APIBase("torch.sparse.sampled_addmm")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.eye(3, device='cuda').to_sparse_csr()
        mat1 = torch.randn(3, 5, device='cuda')
        mat2 = torch.randn(5, 3, device='cuda')
        result = torch.sparse.sampled_addmm(input, mat1, mat2)
        result = result.to_dense()
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        unsupport=True,
        reason="paddle does not support this function temporarily",
    )


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.eye(3, device='cuda').to_sparse_csr()
        mat1 = torch.randn(3, 5, device='cuda')
        mat2 = torch.randn(5, 3, device='cuda')
        result = torch.sparse.sampled_addmm(input=input, mat1=mat1, mat2=mat2)
        result = result.to_dense()
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        unsupport=True,
        reason="paddle does not support this function temporarily",
    )


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.eye(3, device='cuda').to_sparse_csr()
        mat1 = torch.randn(3, 5, device='cuda')
        mat2 = torch.randn(5, 3, device='cuda')
        result = torch.sparse.sampled_addmm(mat1=mat1, mat2=mat2, input=input)
        result = result.to_dense()
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        unsupport=True,
        reason="paddle does not support this function temporarily",
    )


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.eye(3, device='cuda').to_sparse_csr()
        mat1 = torch.randn(3, 5, device='cuda')
        mat2 = torch.randn(5, 3, device='cuda')
        result = torch.sparse.sampled_addmm(input=input, mat1=mat1, mat2=mat2, beta=0.5, alpha=0.5)
        result = result.to_dense()
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        unsupport=True,
        reason="paddle does not support this function temporarily",
    )


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.eye(3, device='cuda').to_sparse_csr()
        mat1 = torch.randn(3, 5, device='cuda')
        mat2 = torch.randn(5, 3, device='cuda')
        out = torch.tensor([])
        result = torch.sparse.sampled_addmm(input=input, mat1=mat1, mat2=mat2, beta=0.5, alpha=0.5, out=out)
        result = result.to_dense()
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        unsupport=True,
        reason="paddle does not support this function temporarily",
    )
