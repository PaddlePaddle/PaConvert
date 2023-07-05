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

obj = APIBase("torch.searchsorted")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[ 1,  3,  5,  7,  9],
                          [ 2,  4,  6,  8, 10]])
        values = torch.tensor([[3, 6, 9],
                               [3, 6, 9]])
        result = torch.searchsorted(x, values)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[ 1,  3,  5,  7,  9],
                          [ 2,  4,  6,  8, 10]])
        values = torch.tensor([[3, 6, 9],
                               [3, 6, 9]])
        result = torch.searchsorted(x, values, out_int32 = True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[ 1,  3,  5,  7,  9],
                          [ 2,  4,  6,  8, 10]])
        values = torch.tensor([[3, 6, 9],
                               [3, 6, 9]])
        result = torch.searchsorted(x, values, right = True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[ 1,  3,  5,  7,  9],
                          [ 2,  4,  6,  8, 10]])
        values = torch.tensor([[3, 6, 9],
                               [3, 6, 9]])
        result = torch.searchsorted(x, values, side = 'right')
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[ 1,  3,  5,  7,  9],
                          [ 2,  4,  6,  8, 10]])
        values = torch.tensor([[3, 6, 9],
                               [3, 6, 9]])
        out = torch.tensor([[3, 6, 9],
                            [3, 6, 9]])
        result = torch.searchsorted(x, values, out = out)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[ 1,  3,  9,  7,  5],
                          [ 2,  4,  6,  8, 10]])
        values = torch.tensor([[3, 6, 9],
                               [3, 6, 9]])
        sorter = torch.argsort(x)
        result = torch.searchsorted(x, values, sorter = sorter)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[ 1,  3,  9,  7,  5],
                          [ 2,  4,  6,  8, 10]])
        values = torch.tensor([[3, 6, 9],
                               [3, 6, 9]])
        out = torch.tensor([[3, 6, 9],
                            [3, 6, 9]])
        sorter = torch.argsort(x)
        result = torch.searchsorted(x, values, right = True, side = 'right', out = out, sorter = sorter)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[ 1,  3,  5,  7,  9],
                          [ 2,  4,  6,  8, 10]])
        values = torch.tensor([[3, 6, 9],
                               [3, 6, 9]])
        result = torch.searchsorted(x, values, right = False, side = 'right')
        """
    )
    obj.run(pytorch_code, ["result"])
