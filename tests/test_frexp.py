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

obj = APIBase("torch.frexp")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([10.0, -2.5, 0.0, 3.14])
        result, exponent = torch.frexp(x)
        """
    )
    obj.run(pytorch_code, ["result", "exponent"], check_dtype=False)


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[128.0, 64.0], [-32.0, 16.0]])
        result, ex = torch.frexp(x)
        """
    )
    obj.run(pytorch_code, ["result", "ex"], check_dtype=False)
