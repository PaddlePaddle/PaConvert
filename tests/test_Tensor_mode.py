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

obj = APIBase("torch.Tensor.mode")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[[1,2,2],[2,3,3]],[[0,5,5],[9,9,0]]])
        result, index = input.mode()
        """
    )
    obj.run(pytorch_code, ["result", "index"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[[1,2,2],[2,3,3]],[[0,5,5],[9,9,0]]])
        result = input.mode()
        result, index = result[0], result[1]
        """
    )
    obj.run(pytorch_code, ["result", "index"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[[1,2,2],[2,3,3]],[[0,5,5],[9,9,0]]])
        result, index = input.mode(1)
        """
    )
    obj.run(pytorch_code, ["result", "index"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[[1,2,2],[2,3,3]],[[0,5,5],[9,9,0]]])
        result, index = input.mode(1, keepdim=True)
        """
    )
    obj.run(pytorch_code, ["result", "index"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[[1,2,2],[2,3,3]],[[0,5,5],[9,9,0]]])
        keepdim = True
        result, index = input.mode(dim=1, keepdim=keepdim)
        """
    )
    obj.run(pytorch_code, ["result", "index"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[[1,2,2],[2,3,3]],[[0,5,5],[9,9,0]]])
        keepdim = True
        result, index = input.mode(1, keepdim)
        """
    )
    obj.run(pytorch_code, ["result", "index"])


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[[1,2,2],[2,3,3]],[[0,5,5],[9,9,0]]])
        keepdim = True
        result, index = input.mode(keepdim=keepdim, dim=1)
        """
    )
    obj.run(pytorch_code, ["result", "index"])
