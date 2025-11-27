# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

obj = APIBase("torch.Tensor.split_with_sizes")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.arange(10)
        split_sizes = [3, 2, 5]
        result = a.split_with_sizes(split_sizes, dim=0)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.arange(6).reshape(2, 3)
        split_sizes = [1, 2]
        result = a.split_with_sizes(split_sizes, dim=1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.arange(4)
        split_sizes = [2, 0, 2]
        result = a.split_with_sizes(split_sizes, dim=0)
        """
    )
    obj.run(pytorch_code, ["result"])
