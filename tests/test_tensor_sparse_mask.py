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

obj = APIBase("torch.tensor.sparse_mask")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        dense = torch.tensor([[1, 2, 3], [4, 5, 6]])
        indices = torch.tensor([[0, 1], [1, 2]])
        values = torch.tensor([0, 0])
        mask = torch.sparse_coo_tensor(indices, values, dense.size())
        result_sparse = dense.sparse_mask(mask)
        result = result_sparse.to_dense()
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        dense = torch.tensor([[5, 0, 3], [0, 8, 2]])
        indices = torch.tensor([[0, 1], [1, 0]])   # 提取 (0,1)=0 和 (1,0)=0
        values = torch.tensor([1, 1])
        mask = torch.sparse_coo_tensor(indices, values, dense.size())
        result_sparse = dense.sparse_mask(mask)
        result = result_sparse.to_dense()
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        dense = torch.arange(24).reshape(2, 3, 4)
        indices = torch.tensor([[0, 1], [1, 2], [3, 0]])  # 提取两个位置
        values = torch.tensor([-1, -1])
        mask = torch.sparse_coo_tensor(indices, values, dense.size())
        result_sparse = dense.sparse_mask(mask)
        result = result_sparse.to_dense()
        """
    )
    obj.run(pytorch_code, ["result"])
  
