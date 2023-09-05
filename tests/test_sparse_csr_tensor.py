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

obj = APIBase("torch.sparse_csr_tensor")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        crows = [0, 2, 3, 5]
        cols = [1, 3, 2, 0, 1]
        values = [1, 2, 3, 4, 5]
        dense_shape = [3, 4]
        result = torch.sparse_csr_tensor(crows, cols, values, dense_shape, requires_grad=False)
        result = result.to_dense()
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        crows = [0, 2, 3, 5]
        cols = [1, 3, 2, 0, 1]
        values = [1, 2, 3, 4, 5]
        dense_shape = [3, 4]
        result = torch.sparse_csr_tensor(crow_indices=crows, col_indices=cols, values=values, size=dense_shape, requires_grad=False)
        result = result.to_dense()
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        crows = [0, 2, 3, 5]
        cols = [1, 3, 2, 0, 1]
        values = [1, 2, 3, 4, 5]
        dense_shape = [3, 4]
        result = torch.sparse_csr_tensor(values=values, size=dense_shape, crow_indices=crows, col_indices=cols, requires_grad=False)
        result = result.to_dense()
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        crows = [0, 2, 3, 5]
        cols = [1, 3, 2, 0, 1]
        values = [1, 2, 3, 4, 5]
        dense_shape = [3, 4]
        result = torch.sparse_csr_tensor(crows, cols, values, dense_shape,
                                         dtype=torch.float32, device=torch.device("cpu"), requires_grad=False, check_invariants=None)
        result = result.to_dense()
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        crows = [0, 2, 3, 5]
        cols = [1, 3, 2, 0, 1]
        values = [1, 2, 3, 4, 5]
        dense_shape = [3, 4]
        result = torch.sparse_csr_tensor(crow_indices=crows, col_indices=cols, values=values, size=dense_shape,
                                         dtype=torch.float32, device=None, requires_grad=False, check_invariants=None)
        result = result.to_dense()
        """
    )
    obj.run(pytorch_code, ["result"])
