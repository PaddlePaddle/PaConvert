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

obj = APIBase("torch.sparse.sum")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        indices = [[0, 1, 2], [1, 2, 0]]
        values = [1.0, 2.0, 3.0]
        dense_shape = [3, 3]
        coo = torch.sparse_coo_tensor(indices, values, dense_shape)
        result = torch.sparse.sum(coo, dim=1).values()
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        indices = [[0, 1, 2], [1, 2, 0]]
        values = [1.0, 2.0, 3.0]
        dense_shape = [3, 3]
        coo = torch.sparse_coo_tensor(indices, values, dense_shape)
        result = torch.sparse.sum(input=coo, dim=-1).values()
        """
    )
    obj.run(pytorch_code, ["result"])


# Torch returns 0-dimensional tensor, while paddle returns sparse tensor.
def _test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        indices = [[0, 1, 2], [1, 2, 0]]
        values = [1.0, 2.0, 3.0]
        dense_shape = [3, 3]
        coo = torch.sparse_coo_tensor(indices, values, dense_shape)
        result = torch.sparse.sum(coo)
        """
    )
    obj.run(pytorch_code, ["result"])
