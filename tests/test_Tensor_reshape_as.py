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

obj = APIBase("torch.Tensor.reshape_as")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.ones([15])
        b = torch.zeros([3, 5])
        result = a.reshape_as(b)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.ones([15])
        b = torch.zeros([3, 5])
        result = a.reshape_as(other=b)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.ones([15])
        b = torch.zeros([3, 5])
        result = a.reshape_as(other=b+1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    # reshape flat tensor into 3-D shape borrowed from another tensor
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.arange(24, dtype=torch.float32)
        template = torch.empty(2, 3, 4)
        result = a.reshape_as(template)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    # integer tensors of different rank
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.arange(12, dtype=torch.int64)
        b = torch.arange(60).reshape(5, 12).to(torch.int64)[0]
        result = a.reshape_as(b)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    # float64 source and target with same total elements but different shapes
    pytorch_code = textwrap.dedent(
        """
        import torch
        src = torch.linspace(-1.0, 1.0, steps=20, dtype=torch.float64)
        ref = torch.randn(4, 5, dtype=torch.float64) * 2 - 1
        result = src.reshape_as(ref)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    # chained: result depends on the reshaped output being used further
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1., 2., 3., 4.]])
        y = torch.tensor([[10., 20.], [30., 40.]])
        r = x.reshape_as(y)
        result = (r + y).sum()
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.arange(24, dtype=torch.float64)
        b = torch.zeros([2, 3, 4], dtype=torch.float32)
        result = a.reshape_as(b)
        """
    )
    obj.run(pytorch_code, ["result"])
