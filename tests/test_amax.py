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

obj = APIBase("torch.amax")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 2, 3], [3, 4, 6]])
        result = torch.amax(x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 2, 3], [3, 4, 6]])
        result = torch.amax(x, dim=1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 2, 3], [3, 4, 6]])
        result = torch.amax(x, 1, True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 2, 3], [3, 4, 6]])
        result = torch.amax(x, dim=0, keepdim=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.amax(torch.tensor([[1, 2, 3], [3, 4, 6]]))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 2, 1], [3, 4, 6]])
        out = torch.tensor([1, 3])
        torch.amax(x, dim=1, keepdim=False, out=out)
        """
    )
    obj.run(pytorch_code, ["out"])


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        dim, keepdim = 1, False
        result = torch.amax(torch.tensor([[1, 2, 3], [3, 4, 6]]), dim, keepdim)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        dim, keepdim = 1, False
        out = torch.tensor([1, 3])
        torch.amax(input=torch.tensor([[1, 2, 3], [3, 4, 6]]), dim=dim, keepdim=keepdim, out=out)
        """
    )
    obj.run(pytorch_code, ["out"])


def test_case_9():
    pytorch_code = textwrap.dedent(
        """
        import torch
        dim, keepdim = 1, False
        out = torch.tensor([1, 3])
        torch.amax(dim=dim, out=out, keepdim=keepdim, input=torch.tensor([[1, 2, 3], [3, 4, 6]]))
        """
    )
    obj.run(pytorch_code, ["out"])


