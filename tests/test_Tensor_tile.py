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


# the only corner case, input a variable which is Constant, has no solution
def _test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.Tensor([1., 2., 3., 4.])
        dims = 1
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
