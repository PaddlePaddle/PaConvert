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

obj = APIBase("torch.Tensor.slice_scatter")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.zeros(8, 8)
        y = torch.ones(8, 8)
        result = x.slice_scatter(y)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.zeros(8, 8)
        y = torch.ones(8, 8)
        result = x.slice_scatter(y, dim=1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.zeros(8, 8)
        y = torch.ones(6, 8)
        result = x.slice_scatter(y, end=6)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.zeros(8, 8)
        y = torch.ones(8, 6)
        result = x.slice_scatter(y, 1, end=6)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.zeros(8, 8)
        y = torch.ones(8, 2)
        result = x.slice_scatter(y, dim=1, start=2, end=6, step=2)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.zeros(8, 8)
        y = torch.ones(8, 2)
        result = x.slice_scatter(src=y, dim=1, start=2, end=6, step=2)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.zeros(8, 8)
        b = torch.ones(8, 2)
        result = a.slice_scatter(dim=1, src=b, step=2, start=2, end=6)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.zeros(8, 8)
        b = torch.ones(8, 2)
        result = a.slice_scatter(b, 1, 2, 6, 2)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.zeros(8, 8)
        b = torch.ones(8, 6)
        result = a.slice_scatter(dim=1, src=b, start=2)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_10():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.zeros(8, 8)
        b = torch.ones(8, 6)
        result = a.slice_scatter(dim=1, src=b, end=6)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_11():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.zeros(8, 8)
        b = torch.ones(8, 8)
        result = a.slice_scatter(src=b)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_12():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.zeros(8, 8)
        b = torch.ones(6, 8)
        result = a.slice_scatter(src=b, start=2)
        """
    )
    obj.run(pytorch_code, ["result"])
