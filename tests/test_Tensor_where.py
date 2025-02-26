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

obj = APIBase("torch.Tensor.where")


# when y is a float scalar, paddle.where will return float64
def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.Tensor([[-1., 2.], [-3., 4.]])
        b = 0.
        result = a.where(a > 0, b)
        """
    )
    obj.run(pytorch_code, ["result"], check_dtype=False)


# paddle.where not support type promotion between x and y, while torch.where support
def _test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.Tensor([[-1., 2.], [-3., 4.]])
        b = 0
        result = a.where(a > 0, 0)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.Tensor([[-1., 2.], [-3., 4.]])
        b = torch.tensor(0.)
        result = a.where(a > 0, b)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.Tensor([[1.,2.], [3.,4.]])
        b = torch.tensor([[0., 0.], [0., 0.]])
        result = a.where(a>0, other=b)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.Tensor([[1., 2.], [3., 4.]])
        b = torch.tensor([[0., 0.], [0., 0.]])
        result = a.where(condition = a>0, other = b)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.Tensor([[1.,2.], [3.,4.]])
        b = torch.tensor([[0., 0.], [0., 0.]])
        result = a.where(other=b, condition=a>0)
        """
    )
    obj.run(pytorch_code, ["result"])
