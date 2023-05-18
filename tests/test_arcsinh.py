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

obj = APIBase("torch.arcsinh")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.arcsinh(torch.tensor([1.3192, 1.9915, 1.9674, 1.7151]))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([1.3192, 1.9915, 1.9674, 1.7151])
        result = torch.arcsinh(a)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = [1.3192, 1.9915, 1.9674, 1.7151]
        out = torch.tensor(a)
        result = torch.arcsinh(torch.tensor(a), out=out)
        """
    )
    obj.run(pytorch_code, ["out", "result"])
