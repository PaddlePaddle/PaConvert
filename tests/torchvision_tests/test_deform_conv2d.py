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

obj = APIBase("torchvision.ops.deform_conv2d")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torchvision.ops import deform_conv2d
        input = torch.tensor([[[[1.0] * 32] * 32] * 3] * 2, dtype=torch.float)
        offset = torch.tensor([[[[0.0] * 30] * 30] * 18] * 2, dtype=torch.float)  # 18 = 2 * 3 * 3
        weight = torch.tensor([[[[1.0] * 3] * 3] * 3] * 4, dtype=torch.float)
        result = deform_conv2d(input=input, offset=offset, weight=weight, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1), mask=None)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torchvision.ops import deform_conv2d
        input = torch.tensor([[[[1.0] * 32] * 32] * 3] * 2, dtype=torch.float)
        offset = torch.tensor([[[[0.0] * 30] * 30] * 18] * 2, dtype=torch.float)  # 18 = 2 * 3 * 3
        weight = torch.tensor([[[[1.0] * 3] * 3] * 3] * 4, dtype=torch.float)
        result = deform_conv2d(input, offset, weight)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torchvision.ops import deform_conv2d
        input = torch.tensor([[[[1.0] * 32] * 32] * 3] * 2, dtype=torch.float)
        offset = torch.tensor([[[[0.0] * 15] * 15] * 18] * 2, dtype=torch.float)  # 18 = 2 * 3 * 3
        weight = torch.tensor([[[[1.0] * 3] * 3] * 3] * 4, dtype=torch.float)
        bias = torch.ones(4, dtype=torch.float)
        result = deform_conv2d(weight=weight, input=input, offset=offset, bias=bias, dilation=(2, 2), stride=(2, 2), padding=(1, 1))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torchvision.ops import deform_conv2d
        input = torch.tensor([[[[1.0] * 32] * 32] * 3] * 2, dtype=torch.float)
        offset = torch.tensor([[[[0.0] * 15] * 15] * 18] * 2, dtype=torch.float)  # 18 = 2 * 3 * 3
        weight = torch.tensor([[[[1.0] * 3] * 3] * 3] * 4, dtype=torch.float)
        mask = torch.tensor([[[[1.0] * 15] * 15] * 9] * 2, dtype=torch.float)  # 9 = 3 * 3
        result = deform_conv2d(input, offset, weight, mask=mask, stride=(2, 2))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torchvision.ops import deform_conv2d
        input = torch.tensor([[[[1.0] * 32] * 32] * 3] * 2, dtype=torch.float)
        offset = torch.tensor([[[[0.0] * 16] * 16] * 18] * 2, dtype=torch.float)  # 18 = 2 * 3 * 3
        weight = torch.tensor([[[[1.0] * 3] * 3] * 3] * 4, dtype=torch.float)
        bias = torch.ones(4, dtype=torch.float)
        mask = torch.tensor([[[[1.0] * 16] * 16] * 9] * 2, dtype=torch.float)  # 9 = 3 * 3
        result = deform_conv2d(mask=mask, offset=offset, input=input, weight=weight, bias=bias, stride=(2, 2), padding=(2, 2), dilation=(2, 2))
        """
    )
    obj.run(pytorch_code, ["result"])
