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

obj = APIBase("torch.tile")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1, 2, 3])
        result = torch.tile(x, (2,))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 2], [0, 6]])
        result = torch.tile(x, (2, 3))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.arange(1, 385, dtype=torch.float32).reshape(8, 6, 4, 2)
        result = torch.tile(x, (2, 2))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, 2.5], [3.0, 4.5], [5.0, 6.5], [7.0, 8.5]])
        result = torch.tile(x, dims=(3, 3, 2, 2))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 2], [3, 5], [7, 11], [13, 17]], dtype=torch.int64)
        dim = (3, 3, 2, 2)
        result = torch.tile(x, dims=dim)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor(
            [[1.25, 2.5], [3.75, 4.5], [5.25, 6.5], [7.75, 8.5]],
            dtype=torch.float64,
        )
        dim = (3, 3, 2, 2)
        result = torch.tile(input=x, dims=dim)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor(
            [[1.5, 2.5], [3.5, 4.5], [5.5, 6.5], [7.5, 8.5]],
            dtype=torch.float32,
        )
        dim = (3, 3, 2, 2)
        result = torch.tile(dims=dim, input=x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        args = (x, (2, 1))
        result = torch.tile(*args)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor(
            [
                [[1, 2], [3, 4]],
                [[5, 6], [7, 8]],
            ],
            dtype=torch.int32,
        )
        result = torch.tile(x, dims=[2, 1, 2])
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_10():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.25, 2.5], [3.75, 4.5]], dtype=torch.float64)
        result = torch.tile(x, dims=(2, 1))
        """
    )
    obj.run(pytorch_code, ["result"])
