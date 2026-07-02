# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

obj = APIBase("torch.nn.init.sparse_")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        tensor = torch.empty(3, 5)
        torch.nn.init.sparse_(tensor, sparsity=0.1, std=0.01)
        result = tensor
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        tensor = torch.empty(4, 6)
        torch.nn.init.sparse_(tensor=tensor, sparsity=0.2, std=0.05)
        result = tensor
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        tensor = torch.empty(5, 10)
        torch.nn.init.sparse_(tensor, std=0.1, sparsity=0.3)
        result = tensor
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        tensor = torch.empty(8, 8)
        torch.nn.init.sparse_(tensor, 0.5, 0.2)
        result = tensor
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        tensor = torch.empty(3, 2)
        torch.nn.init.sparse_(sparsity=0.2, std=0.05, tensor=tensor)
        result = tensor
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)
