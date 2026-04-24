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

obj = APIBase("torch.Tensor.tile")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1., 2., 3., 4.])
        result = x.tile(1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1., 2.], [ 3., 4.]])
        result = x.tile(2, 1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.Tensor([[1., 2.], [3., 4.]])
        result = x.tile((2, 1))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.Tensor([[1., 2.], [3., 4.]])
        result = x.tile([2, 1])
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1., 2., 3., 4.], dtype=torch.float64)
        dims = 2
        result = x.tile(dims)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.Tensor([[1., 2.], [3., 4.]])
        dims = (2, 1)
        result = x.tile(dims)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.Tensor([[1., 2.], [3., 4.]])
        dims = (2, 1)
        result = x.tile(*dims)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.Tensor([[1., 2.], [3., 4.]])
        dims = (2, 1)
        result = x.tile(dims=dims)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.Tensor([[1., 2.], [3., 4.]])
        result = x.tile(dims=(2, 1))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_10():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 3], [5, 7]], dtype=torch.int64)
        dims = (2, 1)
        result = x.tile(*dims)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_11():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor(
            [
                [[1.25, 2.5], [3.75, 4.5]],
                [[5.25, 6.5], [7.75, 8.5]],
            ],
            dtype=torch.float64,
        )
        result = x.tile((2, 1, 2))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_12():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.5, 2.5], [3.5, 4.5]], dtype=torch.float32)
        dims = [2, 3]
        result = x.tile(dims)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_13():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[2.0, 4.0], [6.0, 8.0]], dtype=torch.float32)
        result = x.tile(dims=[1, 2])
        """
    )
    obj.run(pytorch_code, ["result"])
