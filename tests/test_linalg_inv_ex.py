# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

obj = APIBase("torch.linalg.inv_ex")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[0.02773777, 0.93004224, 0.06911496],
                [0.24831591, 0.45733623, 0.07717843],
                [0.48016702, 0.14235102, 0.42620817]])
        out, info = torch.linalg.inv_ex(x)
        """
    )
    obj.run(pytorch_code, ["out", "info"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[0.02773777, 0.93004224, 0.06911496],
                [0.24831591, 0.45733623, 0.07717843],
                [0.48016702, 0.14235102, 0.42620817]])
        out, info = torch.linalg.inv_ex(A=x)
        """
    )
    obj.run(pytorch_code, ["out", "info"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[0.02773777, 0.93004224, 0.06911496],
                [0.24831591, 0.45733623, 0.07717843],
                [0.48016702, 0.14235102, 0.42620817]])
        out1 = torch.tensor([])
        info1 = torch.tensor([1, 2, 3], dtype=torch.int32)
        out1, info1 = torch.linalg.inv_ex(x, out=(out1, info1))
        """
    )
    obj.run(pytorch_code, ["out1", "info1"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[0.02773777, 0.93004224, 0.06911496],
                [0.24831591, 0.45733623, 0.07717843],
                [0.48016702, 0.14235102, 0.42620817]])
        out1 = torch.tensor([])
        info1 = torch.tensor([1, 2, 3], dtype=torch.int32)
        out1, info1 = torch.linalg.inv_ex(x, check_errors=False, out=(out1, info1))
        """
    )
    obj.run(
        pytorch_code,
        ["out1", "info1"],
        unsupport=True,
        reason="paddle does not support this function temporarily",
    )
