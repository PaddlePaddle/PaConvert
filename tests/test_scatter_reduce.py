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

obj = APIBase("torch.scatter_reduce")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        src = torch.tensor([1., 2., 3., 4., 5., 6.])
        index = torch.tensor([0, 1, 0, 1, 2, 1])
        input = torch.tensor([1., 2., 3., 4.])
        result = torch.scatter_reduce(input, 0, index, src, reduce="sum")
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
        src = torch.tensor([1., 2., 3., 4., 5., 6.])
        index = torch.tensor([0, 1, 0, 1, 2, 1])
        input = torch.tensor([1., 2., 3., 4.])
        result = torch.scatter_reduce(input, 0, index, src, reduce="sum", include_self=False)
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
        src = torch.tensor([1., 2., 3., 4., 5., 6.])
        index = torch.tensor([0, 1, 0, 1, 2, 1])
        input = torch.tensor([1., 2., 3., 4.])
        result = torch.scatter_reduce(input, 0, index, src, reduce="amax")
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        unsupport=True,
        reason="paddle does not support this function temporarily",
    )
