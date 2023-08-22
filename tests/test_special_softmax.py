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

obj = APIBase("torch.special.softmax")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[[2.0, 3.0, 4.0, 5.0],
                        [3.0, 4.0, 5.0, 6.0],
                        [7.0, 8.0, 8.0, 9.0]],
                        [[1.0, 2.0, 3.0, 4.0],
                        [5.0, 6.0, 7.0, 8.0],
                        [6.0, 7.0, 8.0, 9.0]]])
        result = torch.special.softmax(x, -1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[[2.0, 3.0, 4.0, 5.0],
                        [3.0, 4.0, 5.0, 6.0],
                        [7.0, 8.0, 8.0, 9.0]],
                        [[1.0, 2.0, 3.0, 4.0],
                        [5.0, 6.0, 7.0, 8.0],
                        [6.0, 7.0, 8.0, 9.0]]])
        result = torch.special.softmax(x, dim=1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[[2.0, 3.0, 4.0, 5.0],
                        [3.0, 4.0, 5.0, 6.0],
                        [7.0, 8.0, 8.0, 9.0]],
                        [[1.0, 2.0, 3.0, 4.0],
                        [5.0, 6.0, 7.0, 8.0],
                        [6.0, 7.0, 8.0, 9.0]]])
        result = torch.special.softmax(x)
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        unsupport=True,
        reason="When dim is None, paddle and pytorch generate different results due to the way to calculate dimensions",
    )


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[[2.0, 3.0, 4.0, 5.0],
                        [3.0, 4.0, 5.0, 6.0],
                        [7.0, 8.0, 8.0, 9.0]],
                        [[1.0, 2.0, 3.0, 4.0],
                        [5.0, 6.0, 7.0, 8.0],
                        [6.0, 7.0, 8.0, 9.0]]])
        result = torch.special.softmax(x, 1, dtype=torch.float64)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[[2.0, 3.0, 4.0, 5.0],
                        [3.0, 4.0, 5.0, 6.0],
                        [7.0, 8.0, 8.0, 9.0]],
                        [[1.0, 2.0, 3.0, 4.0],
                        [5.0, 6.0, 7.0, 8.0],
                        [6.0, 7.0, 8.0, 9.0]]])
        result = torch.special.softmax(input=x, dim=1, dtype=torch.float64)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.special as F
        x = torch.tensor([[[2.0, 3.0, 4.0, 5.0],
                        [3.0, 4.0, 5.0, 6.0],
                        [7.0, 8.0, 8.0, 9.0]],
                        [[1.0, 2.0, 3.0, 4.0],
                        [5.0, 6.0, 7.0, 8.0],
                        [6.0, 7.0, 8.0, 9.0]]])
        result = torch.special.softmax(dim=1, dtype=torch.float64, input=x)
        """
    )
    obj.run(pytorch_code, ["result"])
