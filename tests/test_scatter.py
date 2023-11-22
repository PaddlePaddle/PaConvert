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

obj = APIBase("torch.scatter")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.arange(15).reshape([3, 5]).type(torch.float32)
        index = torch.tensor([[0], [1], [2]])
        result = torch.scatter(input, 1, index, 1.0)
        """
    )
    obj.run(pytorch_code, ["result"])


# paddle broadcast indices && values to arr, while pytorch not
def _test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.arange(15).reshape([3, 5]).type(torch.float32)
        index = torch.tensor([[0, 1, 2]])
        result = torch.scatter(input, 1, index, torch.full([1, 3], -1.), reduce='multiply')
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.arange(15).reshape([3, 5]).type(torch.float32)
        index = torch.tensor([[0], [1], [2]])
        result = torch.scatter(input, 1, index, 1.0, reduce='multiply')
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.arange(15).reshape([3, 5]).type(torch.float32)
        index = torch.tensor([[0], [1], [2]])
        result = torch.scatter(input, 1, index, 1.0, reduce='add')
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.arange(15).reshape([3, 5]).type(torch.float32)
        index = torch.tensor([[0], [1], [2]])
        out = torch.zeros(3, 5)
        result = torch.scatter(input, 1, index, 1.0, reduce='add', out=out)
        """
    )
    obj.run(pytorch_code, ["out"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.arange(15).reshape([3, 5]).type(torch.float32)
        index = torch.tensor([[0, 1, 2], [3, 0, 1], [1, 2, 4]])
        result = torch.scatter(x, 1, index, src=torch.full([3, 3], -1.), reduce='multiply')
        """
    )
    obj.run(pytorch_code, ["result"])
