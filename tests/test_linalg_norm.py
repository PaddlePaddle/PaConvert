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

obj = APIBase("torch.linalg.norm")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        y = torch.tensor([[3, 4, 6], [5, 3, 4], [1, 2, 1.]])
        result = torch.linalg.norm(y, dim=-1, keepdim=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[10, 2, 3], [3, 10, 5], [5, 6, 12.]])
        result = torch.linalg.norm(x, float('inf'), dim=-1)
        """
    )
    obj.run(pytorch_code, ["result"])
