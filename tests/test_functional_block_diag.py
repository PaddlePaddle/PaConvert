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

obj = APIBase("torch.functional.block_diag")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        A = torch.tensor([[0, 1], [1, 0]])
        B = torch.tensor([[3, 4, 5], [6, 7, 8]])
        C = torch.tensor(7)
        D = torch.tensor([1, 2, 3])
        E = torch.tensor([[4], [5], [6]])
        result = torch.functional.block_diag(A, B, C, D, E)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        A = torch.tensor([[4], [3], [2]])
        B = torch.tensor([7, 6, 5])
        C = torch.tensor(1)
        result = torch.functional.block_diag(torch.tensor([[4], [3], [2]]),
                                torch.tensor([7, 6, 5]),
                                torch.tensor(1))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        A = torch.tensor([[4], [3], [2]])
        B = torch.tensor([[5, 6], [9, 1]])
        C = torch.tensor([1, 2, 3])
        result = torch.functional.block_diag(A)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        A = torch.tensor([[4], [3], [2]])
        B = torch.tensor([[5], [6]])
        result = torch.functional.block_diag(A, B, torch.tensor([1, 2, 3]))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        tensors = torch.tensor([[0,1,2]]), torch.tensor([[0],[1]]), torch.tensor([[20]])
        result = torch.functional.block_diag(*tensors)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.functional.block_diag(torch.tensor([[4], [3], [2]]))
        """
    )
    obj.run(pytorch_code, ["result"])
