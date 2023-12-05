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

obj = APIBase("torch.topk")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1, 2, 3, 4, 5])
        result, index = torch.topk(x, 3)
        """
    )
    obj.run(pytorch_code, ["result", "index"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1, 2, 3, 4, 5])
        res = torch.topk(x, 3)
        result, index = res[0], res[1]
        """
    )
    obj.run(pytorch_code, ["result", "index"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 2, 3, 4, 5], [2, 5, 6, 2, 3]])
        result, index = torch.topk(x, 3, dim=1, sorted=True)
        """
    )
    obj.run(pytorch_code, ["result", "index"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 2, 3, 4, 5], [2, 5, 6, 2, 3]])
        result, index = torch.topk(x, 3, 1, True)
        """
    )
    obj.run(pytorch_code, ["result", "index"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 2, 3, 4, 5], [2, 5, 6, 2, 3]])
        out = (torch.zeros(2, 3, dtype=torch.long), torch.zeros(2, 3, dtype=torch.long))
        result, index = torch.topk(x, 3, dim=1, out=out)
        """
    )
    obj.run(pytorch_code, ["result", "index", "out"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 2, 3, 4, 5], [2, 5, 6, 2, 3]])
        out = (torch.zeros(2, 3, dtype=torch.long), torch.zeros(2, 3, dtype=torch.long))
        result, index = torch.topk(input=x, k=3, dim=1, largest=True, sorted=True, out=out)
        """
    )
    obj.run(pytorch_code, ["result", "index", "out"])


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 2, 3, 4, 5], [2, 5, 6, 2, 3]])
        out = (torch.zeros(2, 3, dtype=torch.long), torch.zeros(2, 3, dtype=torch.long))
        result, index = torch.topk(largest=True, k=3, out=out, dim=1, sorted=True, input=x)
        """
    )
    obj.run(pytorch_code, ["result", "index", "out"])


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 2, 3, 4, 5], [2, 5, 6, 2, 3]])
        out = (torch.tensor(1), torch.tensor(2))
        result, index = torch.topk(x, 3, 1, True, True)
        """
    )
    obj.run(pytorch_code, ["result", "index", "out"])
