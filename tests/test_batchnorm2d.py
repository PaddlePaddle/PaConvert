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

obj = APIBase("torch.nn.BatchNorm2d")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch.nn as nn
        import torch
        m = torch.nn.BatchNorm2d(5, affine=False)
        input = torch.zeros(2, 5, 4, 4)
        result = m(input)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch.nn as nn
        import torch
        m = torch.nn.BatchNorm2d(5, affine=False, eps=1e-5)
        input = torch.zeros(2, 5, 4, 4)
        result = m(input)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch.nn as nn
        import torch
        m = torch.nn.BatchNorm2d(5, 1e-5, 0.2, affine=False)
        input = torch.zeros(2, 5, 4, 4)
        result = m(input)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch.nn as nn
        import torch
        m = torch.nn.BatchNorm2d(5, 1e-5, 0.2, affine=True)
        input = torch.zeros(2, 5, 4, 4)
        result = m(input)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch.nn as nn
        import torch
        m = torch.nn.BatchNorm2d(5, 1e-5, 0.2, affine=True, track_running_stats=True)
        input = torch.zeros(2, 5, 4, 4)
        result = m(input)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch.nn as nn
        import torch
        m = torch.nn.BatchNorm2d(5)
        input = torch.zeros(2, 5, 4, 4)
        result = m(input)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch.nn as nn
        import torch
        a = False
        m = torch.nn.BatchNorm2d(5, affine=a)
        input = torch.zeros(2, 5, 4, 4)
        result = m(input)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch.nn as nn
        import torch
        a = True
        m = torch.nn.BatchNorm2d(5, 1e-5, 0.2, affine=a, track_running_stats=True)
        input = torch.zeros(2, 5, 4, 4)
        result = m(input)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    pytorch_code = textwrap.dedent(
        """
        import torch.nn as nn
        import torch
        a = True
        m = torch.nn.BatchNorm2d(5, 1e-5, 0.2, affine=a, track_running_stats=True, dtype=torch.float32)
        input = torch.zeros(2, 5, 4, 4)
        result = m(input)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_alias_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch.nn as nn
        import torch
        a = True
        m = torch.nn.modules.BatchNorm2d(5, 1e-5, 0.2, affine=a, track_running_stats=True, dtype=torch.float32)
        input = torch.zeros(2, 5, 4, 4)
        result = m(input)
        """
    )
    obj.run(pytorch_code, ["result"])
