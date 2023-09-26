# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserfed.
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

obj = APIBase("torch.special.erfinv")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.special.erfinv(torch.tensor([0, 0.5, -1.]))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([0, 0.5, -1.])
        result = torch.special.erfinv(input=a)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([0, 0.5, -1.])
        out = torch.tensor([])
        result = torch.special.erfinv(a, out=out)
        """
    )
    obj.run(pytorch_code, ["result", "out"])
