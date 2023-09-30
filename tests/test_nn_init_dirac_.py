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

obj = APIBase("torch.nn.init.dirac_")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        conv = torch.nn.Conv2d(4, 6, (3, 3))
        torch.nn.init.dirac_(conv.weight)
        result = conv.weight
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        conv = torch.nn.Conv2d(3, 6, (3, 3))
        torch.nn.init.dirac_(conv.weight, 1)
        result = conv.weight
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        conv = torch.nn.Conv2d(3, 6, (3, 3))
        torch.nn.init.dirac_(tensor=conv.weight, groups=1)
        result = conv.weight
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        conv = torch.nn.Conv2d(3, 6, (3, 3))
        torch.nn.init.dirac_(groups=1, tensor=conv.weight)
        result = conv.weight
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        conv = torch.nn.Conv2d(3, 6, (3, 3))
        torch.nn.init.dirac_(conv.weight, groups=1)
        result = conv.weight
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        conv = torch.nn.Conv2d(3, 6, (3, 3))
        torch.nn.init.dirac_(tensor=conv.weight)
        result = conv.weight
        """
    )
    obj.run(pytorch_code, ["result"])
