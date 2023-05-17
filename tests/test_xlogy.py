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

obj = APIBase("torch.xlogy")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([1., 2., 3., 4., 5.])
        b = torch.tensor([1., 2., 3., 4., 5.])
        result = torch.xlogy(a, b)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.xlogy(input=torch.tensor([1., 2., 3., 4., 5.]), other=torch.tensor([1., 2., 3., 4., 5.]))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([1., 2., 3., 4., 5.])
        out = torch.tensor([1., 2., 3., 4., 5.])
        result = torch.xlogy(a, a, out=out)
        """
    )
    obj.run(pytorch_code, ["out"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.xlogy(torch.tensor([1.]),torch.tensor([1., 2., 3., 4., 5.]))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.xlogy(1.,torch.tensor([1., 2., 3., 4., 5.]))
        """
    )
    obj.run(pytorch_code, ["result"])


def _test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.xlogy(1, torch.tensor([1, 2, 3, 4, 5]))
        """
    )
    obj.run(pytorch_code, ["result"])


def _test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1., float('inf'), float('nan')])
        result = torch.xlogy(0, x)
        """
    )
    obj.run(pytorch_code, ["result"])


def _test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.xlogy(1, torch.tensor([1., 2., 3., 4., 5.], dtype=torch.float64))
        """
    )
    obj.run(pytorch_code, ["result"])
