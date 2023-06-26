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

obj = APIBase("torch.Tensor.clip_")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([-1.7120,  0.1734, -0.0478, -0.0922])
        result = x.clip_(-0.5, 0.5)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([-1.7120,  0.1734, -0.0478, -0.0922])
        min, max = -0.5, 0.5
        result = x.clip_(min, max)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.tensor([-1.7120,  0.1734, -0.0478, -0.0922]).clip_(min=-0.5, max=0.5)
        """
    )
    obj.run(pytorch_code, ["result"])


# the min param of paddle does not support specifying multiple lower bounds, and by default only the first element is taken.
def _test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        min = torch.linspace(-1, 1, steps=4)
        result = torch.tensor([-1.7120,  0.1734, -0.0478, -0.0922]).clip_(min=min)
        """
    )
    obj.run(pytorch_code, ["result"])


# the max param of paddle does not support specifying multiple upper bounds, and by default only the first element is taken.
def _test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        max = torch.linspace(0, 2, steps=4)
        result = torch.tensor([-1.7120,  0.1734, -0.0478, -0.0922]).clip_(max=max)
        """
    )
    obj.run(pytorch_code, ["result"])
