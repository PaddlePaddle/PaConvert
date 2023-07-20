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
#

import textwrap

from apibase import APIBase

obj = APIBase("torch.scatter_add")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        src = torch.ones((1, 5))
        index = torch.tensor([[0, 1, 2, 0, 0]])
        input = torch.zeros(3, 5, dtype=src.dtype)
        result = torch.scatter_add(input,0, index, src)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        src = torch.ones((2, 5))
        index = torch.tensor([[0, 1, 2, 0, 0], [0, 1, 2, 2, 2]])
        input = torch.zeros(3, 5, dtype=src.dtype)
        result = torch.scatter_add(input,0, index, src)
        """
    )
    obj.run(pytorch_code, ["result"])
