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

obj = APIBase("torch.linalg.qr")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[12., -51, 4], [6, 167, -68], [-4, 24, -41]])
        Q, R = torch.linalg.qr(x)
        """
    )
    obj.run(pytorch_code, ["Q", "R"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[12., -51, 4], [6, 167, -68], [-4, 24, -41]])
        Q, R = torch.linalg.qr(x, mode='reduced')
        """
    )
    obj.run(pytorch_code, ["Q", "R"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[12., -51, 4], [6, 167, -68], [-4, 24, -41]])
        Q, R = torch.linalg.qr(A=x, mode='complete')
        """
    )
    obj.run(pytorch_code, ["Q", "R"])


# when mode='r', torch return empty Q and R, but paddle only return R, should fix Matcher to adapt
def _test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[12., -51, 4], [6, 167, -68], [-4, 24, -41]])
        result = torch.linalg.qr(A=x, mode='r')
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[12., -51, 4], [6, 167, -68], [-4, 24, -41]])
        out = (torch.tensor([]), torch.tensor([]))
        result = torch.linalg.qr(x, out=out)
        """
    )
    obj.run(pytorch_code, ["result", "out"])
