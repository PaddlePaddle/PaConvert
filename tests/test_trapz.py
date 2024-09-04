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

obj = APIBase("torch.trapz")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        y = torch.tensor([1.0, 1, 1, 0, 1])
        result = torch.trapz(y)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        y = torch.tensor([1, 1, 1, 0, 1]).type(torch.float32)
        x = torch.tensor([1, 2, 3, 0, 1]).type(torch.float32)
        result = torch.trapz(y=y, x=x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        y = torch.tensor([1, 1, 1, 0, 1]).type(torch.float32)
        x = torch.tensor([1, 2, 3, 0, 1]).type(torch.float32)
        result = torch.trapz(y=y, dim=-1, dx=2)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        y = torch.tensor([1, 1, 1, 0, 1]).type(torch.float32)
        x = torch.tensor([1, 2, 3, 0, 1]).type(torch.float32)
        result = torch.trapz(y, dx=2, dim=-1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        y = torch.tensor([1, 1, 1, 0, 1]).type(torch.float32)
        x = torch.tensor([1, 2, 3, 0, 1]).type(torch.float32)
        result = torch.trapz(y=y, x=x, dim=-1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        y = torch.tensor([1, 1, 1, 0, 1]).type(torch.float32)
        x = torch.tensor([1, 2, 3, 0, 1]).type(torch.float32)
        result = torch.trapz(dx=2, y=y, dim=-1)
        """
    )
    obj.run(pytorch_code, ["result"])

def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        y = torch.tensor([1, 1, 1, 0, 1]).type(torch.float32)
        x = torch.tensor([1, 2, 3, 0, 1]).type(torch.float32)
        result = torch.trapz(y=y, dx=2, dim=-1)
        """
    )
    obj.run(pytorch_code, ["result"])