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

obj = APIBase("torch.Tensor.index_fill")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.eye(2, 4)
        indices = torch.tensor([0, 1])
        value = -1
        result = x.index_fill(0, indices, value)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        indices = torch.tensor([0, 1])
        value = -1
        result = torch.eye(3, 4).index_fill(1, indices, value)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        indices = torch.tensor([0, 1])
        dim = 0
        value = -1
        result = torch.eye(3, 4).index_fill(index=indices, dim=dim, value=value)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        indices = torch.tensor([0, 3])
        dim = 0
        value = -1
        result = torch.eye(6, 4).index_fill(dim=dim, index=indices, value=value)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        indices = torch.tensor([0, 3])
        value = -1
        result = torch.eye(3, 4).index_fill(1, indices, value)
        """
    )
    obj.run(pytorch_code, ["result"])
