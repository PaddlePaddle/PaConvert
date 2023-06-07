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

obj = APIBase("torch.divide")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([4.0, 3.0])
        b = torch.tensor([2.0, 2.0])
        result = torch.floor_divide(a, b)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.floor_divide(torch.tensor([4.0, 3.0]), torch.tensor([2.0, 2.0]))
        """
    )
    obj.run(pytorch_code, ["result"])


# 'paddle.floor_divide' argument 'y' can only be Tensor, can't be int
def _test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.floor_divide(input=torch.tensor([4.0, 3.0]), other=2)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        out = torch.tensor([4.0, 3.0])
        result = torch.floor_divide(input=torch.tensor([4.0, 3.0]), other=torch.tensor([2.0, 2.0]))
        """
    )
    obj.run(pytorch_code, ["result", "out"])
