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

obj = APIBase("torch.linalg.matrix_norm")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[0.02773777, 0.93004224, 0.06911496],
                [0.24831591, 0.45733623, 0.07717843],
                [0.48016702, 0.14235102, 0.42620817]])
        result = torch.linalg.matrix_norm(x)
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        unsupport=True,
        reason="paddle does not support this function temporarily",
    )


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[0.02773777, 0.93004224, 0.06911496],
                [0.24831591, 0.45733623, 0.07717843],
                [0.48016702, 0.14235102, 0.42620817]])
        result = torch.linalg.matrix_norm(A=x, ord='fro')
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        unsupport=True,
        reason="paddle does not support this function temporarily",
    )


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[0.02773777, 0.93004224, 0.06911496],
                [0.24831591, 0.45733623, 0.07717843],
                [0.48016702, 0.14235102, 0.42620817]])
        out = torch.tensor([])
        result = torch.linalg.matrix_norm(A=x, ord='fro', dtype=torch.float32, out=out)
        """
    )
    obj.run(
        pytorch_code,
        ["result", "out"],
        unsupport=True,
        reason="paddle does not support this function temporarily",
    )
