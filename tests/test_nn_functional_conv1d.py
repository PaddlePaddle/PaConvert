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


class API(APIBase):
    def check(self, pytorch_result, paddle_result):
        torch_numpy, paddle_numpy = pytorch_result.numpy(), paddle_result.numpy()
        if torch_numpy.shape != paddle_numpy.shape:
            return False
        if pytorch_result.requires_grad == paddle_result.stop_gradient:
            return False
        if str(pytorch_result.dtype)[6:] != str(paddle_result.dtype)[7:]:
            return False
        return True


obj = API("torch.nn.functional.conv1d")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.randn(33, 16, 30)
        weight = torch.randn(20, 16, 5)
        result = F.conv1d(x, weight)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.randn(33, 16, 30)
        weight = torch.randn(20, 16, 5)
        bias = torch.randn(20)
        result = F.conv1d(x, weight, bias)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.randn(33, 16, 30)
        weight = torch.randn(20, 16, 5)
        result = F.conv1d(x, weight, stride=2)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.randn(33, 16, 30)
        weight = torch.randn(20, 16, 5)
        result = F.conv1d(x, weight, stride=2, padding=2)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.randn(33, 16, 30)
        weight = torch.randn(20, 16, 5)
        result = F.conv1d(x, weight, stride=2, padding=2, dilation=1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.randn(33, 16, 30)
        weight = torch.randn(20, 8, 5)
        result = F.conv1d(x, weight, stride=2, padding=2, dilation=1, groups=2)
        """
    )
    obj.run(pytorch_code, ["result"])
