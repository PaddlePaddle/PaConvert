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
#

import textwrap

from apibase import APIBase

obj = APIBase("torch.special.erfc")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.special.erfc(torch.tensor([0, -1., 10.]))
        """
    )
    obj.run(pytorch_code, ["result"], atol=1.0e-6)


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([0, -1., 10.])
        result = torch.special.erfc(a)
        """
    )
    obj.run(pytorch_code, ["result"], atol=1.0e-6)


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([0, -1., 10.])
        out = torch.tensor([0, -1., 10.])
        result = torch.special.erfc(a, out=out)
        """
    )
    obj.run(pytorch_code, ["result", "out"], atol=1.0e-6)


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([0, -1., 10.])
        result = torch.special.erfc(input=a)
        """
    )
    obj.run(pytorch_code, ["result"], atol=1.0e-6)


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([0, -1., 10.])
        out = torch.tensor([0, -1., 10.])
        result = torch.special.erfc(input=a, out=out)
        """
    )
    obj.run(pytorch_code, ["result", "out"], atol=1.0e-6)


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([0, -1., 10.])
        out = torch.tensor([0, -1., 10.])
        result = torch.special.erfc(out=out, input=a)
        """
    )
    obj.run(pytorch_code, ["result", "out"], atol=1.0e-6)
