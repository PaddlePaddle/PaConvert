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

obj = APIBase("torch.bernoulli")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([0.8, 0.1, 0.4])
        result = torch.bernoulli(a)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.ones(3, 3)
        result = torch.bernoulli(a)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.bernoulli(torch.ones(3, 3))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.bernoulli(torch.zeros(3, 3))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.ones(3, 3)
        out = torch.zeros(3, 3)
        result = torch.bernoulli(a, out=out)
        """
    )
    obj.run(pytorch_code, ["result", "out"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.rand(3, 3)
        result = torch.bernoulli(a, 1.0)
        """
    )
    obj.run(
        pytorch_code,
        ["a", "result"],
        unsupport=True,
        reason="paddle not support parameter 'p' ",
    )


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.ones(3, 3)
        out = torch.zeros(3, 3)
        result = torch.bernoulli(a, out=out, generator=torch.Generator())
        """
    )
    obj.run(pytorch_code, ["result", "out"])


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.ones(3, 3)
        out = torch.zeros(3, 3)
        result = torch.bernoulli(input=a, generator=torch.Generator(), out=out)
        """
    )
    obj.run(pytorch_code, ["result", "out"])


def test_case_9():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.ones(3, 3)
        out = torch.zeros(3, 3)
        result = torch.bernoulli(generator=torch.Generator(), input=a, out=out)
        """
    )
    obj.run(pytorch_code, ["result", "out"])
