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

obj = APIBase("torch.clamp_max")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([-1.7120,  0.1734, -0.0478, 0.8922])
        result = torch.clamp_max(a, -0.5)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([-1.7120,  0.1734, -0.0478, 0.8922])
        result = torch.clamp_max(a, max=-0.2)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([-1.7120,  0.1734, -0.0478, 0.8922])
        result = torch.clamp_max(input=a, max=-0.5)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([-1.7120,  0.1734, -0.0478, 0.8922])
        out = torch.ones_like(a)
        result = torch.clamp_max(input=a, max=-0.5, out=out)
        """
    )
    obj.run(pytorch_code, ["result", "out"])


# the max param of paddle does not support specifying multiple lower bounds, and by default only the first element is taken.
def _test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([-1.7120,  0.1734, -0.0478, 0.8922])
        max = torch.tensor([-0.3, 0.04, 0.23, 0.98])
        result = torch.clamp_max(a, max=max)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([-1.7120,  0.1734, -0.0478, 0.8922])
        out = torch.ones_like(a)
        result = torch.clamp_max(out=out, input=a, max=-0.5)
        """
    )
    obj.run(pytorch_code, ["result", "out"])
