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

obj = APIBase("torch.histogram")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        bins = 4
        hist, bin = torch.histogram(torch.tensor([[1., 2, 1]]), 4, range=(0., 3.))
        """
    )
    obj.run(pytorch_code, ["hist", "bin"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        bins = 4
        hist, bin = torch.histogram(torch.tensor([[1., 2, 1]]), 4, range=(0., 3.))
        """
    )
    obj.run(pytorch_code, ["hist", "bin"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        bins = 4
        out1 = torch.zeros([bins], dtype=torch.float64)
        out2 = torch.zeros([bins + 1], dtype=torch.float64)
        result = torch.histogram(torch.tensor([[1, 2, 1]], dtype=torch.float64), 4, range=(0., 3.), out=(out1, out2))
        """
    )
    obj.run(pytorch_code, ["out1", "out2"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        bins = 4
        out1 = torch.zeros([bins])
        out2 = torch.zeros([bins + 1])
        result = torch.histogram(torch.tensor([[1., 2, 1]]), 4, range=(0, 3), out=(out1, out2))
        """
    )
    obj.run(pytorch_code, ["out1", "out2"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        bins = 4
        out1 = torch.zeros([bins])
        out2 = torch.zeros([bins + 1])
        weight = torch.tensor([1., 2., 4.])
        result = torch.histogram(torch.tensor([1., 2, 1]), 4, range=(0, 3), weight=weight, out=(out1, out2))
        """
    )
    obj.run(pytorch_code, ["out1", "out2"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        bins = 4
        out1 = torch.zeros([bins])
        out2 = torch.zeros([bins + 1])
        weight = torch.tensor([1., 2., 4.])
        density = True
        result = torch.histogram(torch.tensor([1., 2, 1]), 4, range=(0, 3), weight=weight, density=density, out=(out1, out2))
        """
    )
    obj.run(pytorch_code, ["out1", "out2"])


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        bins = 4
        out1 = torch.zeros([bins])
        out2 = torch.zeros([bins + 1])
        density = True
        result = torch.histogram(torch.tensor([1., 2, 1]), 4, range=(0, 3), density=density, out=(out1, out2))
        """
    )
    obj.run(pytorch_code, ["out1", "out2"])


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        bins = 4
        out1 = torch.zeros([bins], dtype=torch.float64)
        out2 = torch.zeros([bins + 1], dtype=torch.float64)
        weight = torch.tensor([1., 2., 4.], dtype=torch.float64)
        density = True
        result = torch.histogram(torch.tensor([1., 2, 1], dtype=torch.float64), 4, range=(0, 3), weight=weight, density=density, out=(out1, out2))
        """
    )
    obj.run(pytorch_code, ["out1", "out2"])


def test_case_9():
    pytorch_code = textwrap.dedent(
        """
        import torch
        bins = 4
        hist, bin = torch.histogram(torch.tensor([[1., 2, 1]]), bins)
        """
    )
    obj.run(pytorch_code, ["hist", "bin"])


def test_case_10():
    pytorch_code = textwrap.dedent(
        """
        import torch
        bins = 4
        hist, bin = torch.histogram(torch.tensor([[1., 2, 1]]), 4, range=[0, 3])
        """
    )
    obj.run(pytorch_code, ["hist", "bin"])


def test_case_11():
    pytorch_code = textwrap.dedent(
        """
        import torch
        bins = 4
        out1 = torch.zeros([bins], dtype=torch.float64)
        out2 = torch.zeros([bins + 1], dtype=torch.float64)
        weight = torch.tensor([1., 2., 4.], dtype=torch.float64)
        density = True
        result = torch.histogram(input=torch.tensor([1., 2, 1], dtype=torch.float64), bins=4, range=(0, 3), weight=weight, density=density, out=(out1, out2))
        """
    )
    obj.run(pytorch_code, ["out1", "out2"])


def test_case_12():
    pytorch_code = textwrap.dedent(
        """
        import torch
        bins = 4
        out1 = torch.zeros([bins], dtype=torch.float64)
        out2 = torch.zeros([bins + 1], dtype=torch.float64)
        weight = torch.tensor([1., 2., 4.], dtype=torch.float64)
        density = True
        result = torch.histogram(weight=weight, density=density, bins=4, range=(0, 3), input=torch.tensor([1., 2, 1], dtype=torch.float64), out=(out1, out2))
        """
    )
    obj.run(pytorch_code, ["out1", "out2"])


def test_case_13():
    pytorch_code = textwrap.dedent(
        """
        import torch
        bins = 4
        weight = torch.tensor([1., 2., 4.], dtype=torch.float64)
        density = False
        hist, bin = torch.histogram(torch.tensor([1., 2, 1], dtype=torch.float64), 4, range=(0, 3), weight=weight, density=density)
        """
    )
    obj.run(pytorch_code, ["hist", "bin"])
