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


obj = API("torch.nn.functional.fold")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.randn(1, 3 * 2 * 2, 12)
        result = F.fold(x, output_size=(4, 5), kernel_size=(2, 2))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.randn(1, 3 * 2 * 2, 12)
        result = F.fold(x, output_size=(4, 5), kernel_size=(2, 2), stride=1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.randn(1, 3 * 2 * 2, 12)
        result = F.fold(x, output_size=(4, 5), kernel_size=(2, 2), dilation=1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.randn(1, 3 * 2 * 2, 12)
        result = F.fold(x, output_size=(4, 5), kernel_size=(2, 2), dilation=1, padding=0)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.randn(1, 3 * 2 * 2, 12)
        result = F.fold(x, output_size=(4, 5), kernel_size=(2, 2), dilation=1, padding=0, stride=1)
        """
    )
    obj.run(pytorch_code, ["result"])
