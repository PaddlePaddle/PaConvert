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

import paddle
import pytest
from apibase import APIBase

obj = APIBase("torch.sparse.mm")


@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="can only run on paddle with CUDA",
)
def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        indices = [[0, 1, 2], [1, 2, 0]]
        values = [1., 2., 3.]
        x = torch.sparse_coo_tensor(indices, values, [3, 3])
        dense = torch.ones([3, 2])
        result = torch.sparse.mm(x, dense)
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="can only run on paddle with CUDA",
)
def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        indices = [[0, 1, 2], [1, 2, 0]]
        values = [1., 2., 3.]
        x = torch.sparse_coo_tensor(indices, values, [3, 3])
        dense = torch.ones([3, 2])
        result = torch.sparse.mm(sparse=x, dense=dense)
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="can only run on paddle with CUDA",
)
def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        indices = [[0, 1, 2], [1, 2, 0]]
        values = [1., 2., 3.]
        x = torch.sparse_coo_tensor(indices, values, [3, 3])
        dense = torch.ones([3, 2])
        result = torch.sparse.mm(dense=dense, sparse=x)
        """
    )
    obj.run(pytorch_code, ["result"])
