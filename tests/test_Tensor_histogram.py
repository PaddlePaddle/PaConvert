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

obj = APIBase("torch.Tensor.histogram")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        hist, bin = torch.tensor([[1., 2, 1]]).histogram(bins=4, range=(0., 3.))
        """
    )
    obj.run(pytorch_code, ["hist", "bin"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        hist, bin = torch.tensor([[1., 2, 1]]).histogram(bins=4, range=(0., 3.))
        """
    )
    obj.run(pytorch_code, ["hist", "bin"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
        hist, bin = input.histogram(bins=4, range=(0., 3.))
        """
    )
    obj.run(pytorch_code, ["hist", "bin"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
        hist, bin = input.histogram(bins=4, range=[0., 3.])
        """
    )
    obj.run(pytorch_code, ["hist", "bin"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        hist, bin = torch.tensor([[1., 2, 1]]).histogram(10)
        """
    )
    obj.run(pytorch_code, ["hist", "bin"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[1, 2], [3, 4]], dtype=torch.float64)
        hist, bin = input.histogram(bins=4, range=[0., 3.])
        """
    )
    obj.run(pytorch_code, ["hist", "bin"])


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        bins = 4
        input = torch.tensor([1., 2, 1], dtype=torch.float64)
        weight = torch.tensor([1., 2., 4.], dtype=torch.float64)
        density = True
        hist, bin = input.histogram(bins=4, range=[0., 3.], weight=weight, density=density)
        """
    )
    obj.run(pytorch_code, ["hist", "bin"])


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        bins = 4
        input = torch.tensor([1., 2, 1], dtype=torch.float64)
        weight = torch.tensor([1., 2., 4.], dtype=torch.float64)
        density = False
        hist, bin = input.histogram(bins=4, range=[0., 3.], weight=weight, density=density)
        """
    )
    obj.run(pytorch_code, ["hist", "bin"])


def test_case_9():
    pytorch_code = textwrap.dedent(
        """
        import torch
        bins = 4
        input = torch.tensor([1., 2, 1])
        weight = None
        density = True
        hist, bin = input.histogram(bins=4, range=[0., 3.], weight=weight, density=density)
        """
    )
    obj.run(pytorch_code, ["hist", "bin"])


def test_case_10():
    pytorch_code = textwrap.dedent(
        """
        import torch
        hist, bin = torch.tensor([[1., 2, 1]]).histogram(bins=4, range=(0, 3))
        """
    )
    obj.run(pytorch_code, ["hist", "bin"])
