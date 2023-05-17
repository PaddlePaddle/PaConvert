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

obj = APIBase("torch.sort")


def _test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[4, 9], [23, 2]])
        result = torch.sort(a)
        """
    )
    obj.run(pytorch_code, ["result"])


def _test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[4, 9], [23, 2]])
        result = torch.sort(a, 0)
        """
    )
    obj.run(pytorch_code, ["result"])


def _test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[4, 9], [23, 2]])
        result = torch.sort(input=a, dim=1, descending=True, stable=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def _test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        descending=False
        result = torch.sort(torch.tensor([[4, 9], [23, 2]]), dim=1, descending=descending, stable=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def _test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[4, 9], [23, 2]])
        out = torch.tensor(a)
        result = torch.sort(input=a, dim=1, descending=True, stable=True, out=out)
        """
    )
    obj.run(pytorch_code, ["result", "out"])
