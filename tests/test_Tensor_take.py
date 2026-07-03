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

obj = APIBase("torch.Tensor.take")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[4, 3, 5],
                              [6, 7, 8]])
        result = input.take(torch.tensor([0, 2, 5]))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[4, 3, 5],
                              [6, 7, 8]])
        result = input.take(index=torch.tensor([0, 2, 5]))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[4, 3, 5],
                              [6, 7, 8]])
        indices = [0, 2, 5]
        result = input.take(torch.tensor(indices))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import numpy as np
        np.random.seed(42)
        input = torch.from_numpy(np.random.randn(2, 3, 4))
        result = input.take(torch.tensor([0, 5, 10, 15, 20]))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([10, 20, 30])
        result = input.take(torch.tensor([0, 2]))
        """
    )
    obj.run(pytorch_code, ["result"])
