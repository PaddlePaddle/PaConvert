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

obj = APIBase("torch.Tensor.sub_")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([0.5950, -0.0872, 2.3298, -0.2972])
        b = torch.tensor([1., 2., 3., 4.])
        a.sub_(b)
        """
    )
    obj.run(pytorch_code, ["a"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([ 0.5950,-0.0872, 2.3298, -0.2972])
        b = torch.tensor([1., 2., 3., 4.])
        a.sub_(other=b)
        """
    )
    obj.run(pytorch_code, ["a"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([ 0.5950,-0.0872, 2.3298, -0.2972])
        b = torch.tensor([1., 2., 3., 4.])
        a.sub_(b, alpha=3)
        """
    )
    obj.run(pytorch_code, ["a"])


# paddle not support input python number, x/y must be Tensor
def _test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([ 0.5950,-0.0872, 2.3298, -0.2972])
        a.sub_(0.5, alpha=3)
        """
    )
    obj.run(pytorch_code, ["a"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([ 0.5950,-0.0872, 2.3298, -0.2972])
        b = torch.tensor([1., 2., 3., 4.])
        a.sub_(other=b, alpha=3)
        """
    )
    obj.run(pytorch_code, ["a"])


# paddle.sub_ has bug, when float - int, but result'dtype is int, wrong type promote
def _test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.ones((10, 4))
        x.sub_(4, alpha=5)
        """
    )
    obj.run(pytorch_code, ["x"])


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([ 0.5950,-0.0872, 2.3298, -0.2972])
        b = torch.tensor([1., 2., 3., 4.])
        a.sub_(alpha=3, other=b)
        """
    )
    obj.run(pytorch_code, ["a"])
