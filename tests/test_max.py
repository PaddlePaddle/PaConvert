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

obj = APIBase("torch.max", is_aux_api=True)


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 2, 3], [3, 4, 6]])
        out = torch.tensor([1, 2, 3])
        result = torch.max(x)
        """
    )
    obj.run(pytorch_code, ["result", "out"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.max(torch.tensor([[1, 2, 3], [3, 4, 6]]))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 2, 3], [3, 4, 6]])
        result = torch.max(x, 1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 2, 3], [3, 4, 6]])
        result = torch.max(x, -1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 2, 3], [3, 4, 6]])
        result = torch.max(x, dim=-1, keepdim=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 2, 3], [3, 4, 6]])
        result = torch.max(input=x, dim=1, keepdim=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 2, 1], [3, 4, 6]])
        out = [torch.tensor(0), torch.tensor(1)]
        result = torch.max(x, 1, False, out=out)
        """
    )
    obj.run(pytorch_code, ["result", "out"])


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        dim = 1
        keepdim = False
        result = torch.max(torch.tensor([[1, 2, 3], [3, 4, 6]]), dim, keepdim)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.max(torch.tensor([[1, 2, 3], [3, 4, 6]]), torch.tensor([[1, 0, 3], [3, 4, 3]]))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_10():
    pytorch_code = textwrap.dedent(
        """
        import torch
        other = torch.tensor([[1, 0, 3], [3, 4, 3]])
        result = torch.max(torch.tensor([[1, 2, 3], [3, 4, 6]]), other)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_11():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.max(input=torch.tensor([[1, 2, 3], [3, 4, 6]]), other=torch.tensor([[1, 0, 3], [3, 4, 3]]))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_12():
    pytorch_code = textwrap.dedent(
        """
        import torch
        other = torch.tensor([[1, 0, 3], [3, 4, 3]])
        out = torch.tensor([[1, 0, 3], [3, 4, 3]])
        result = torch.max(input=torch.tensor([[1, 2, 3], [3, 4, 6]]), other=other, out=out)
        """
    )
    obj.run(pytorch_code, ["result", "out"])
