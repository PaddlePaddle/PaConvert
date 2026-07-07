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

obj = APIBase("torch.special.sinc")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([ 0.5950,-0.0872, 0, -0.2972])
        result = torch.special.sinc(a)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([ 0.5950,-0.0872, 0, -0.2972])
        result = torch.special.sinc(input=a)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([ 0.5950,-0.0872, 0, -0.2972])
        out = torch.tensor([ 0.5950,-0.0872, 0, -0.2972])
        result = torch.special.sinc(input=a, out=out)
        """
    )
    obj.run(pytorch_code, ["out"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([ 0.5950,-0.0872, 0, -0.2972])
        out = torch.tensor([ 0.5950,-0.0872, 0, -0.2972])
        result = torch.special.sinc(out=out, input=a)
        """
    )
    obj.run(pytorch_code, ["out"])


def test_case_5():
    # two-dimensional float32 input including the zero element
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[1., -2., 3.], [0., -4., 5.]])
        result = torch.special.sinc(input=a)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    # high precision grid covering both integer and fractional inputs
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.linspace(-1.5, 1.5, steps=19, dtype=torch.float64)
        result = torch.special.sinc(a)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    # expression-derived argument passed positionally
    pytorch_code = textwrap.dedent(
        """
        import torch
        n = (torch.arange(9) % 3) * (-0.5 + 0.25*4)
        x = n.to(torch.float64) / 2
        result = torch.special.sinc(x)
        """
    )
    obj.run(pytorch_code, ["result"])
