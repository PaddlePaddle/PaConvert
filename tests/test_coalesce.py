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

obj = APIBase("torch.Tensor.coalesce")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        i = torch.tensor([[0, 1, 1],
                          [2, 0, 2]])
        v = torch.tensor([3, 4, 5], dtype=torch.float32)
        result = torch.sparse_coo_tensor(i, v, [2, 4])
        v = result.coalesce()
        result = result.to_dense()
        """
    )
    obj.run(pytorch_code, ["result"])
