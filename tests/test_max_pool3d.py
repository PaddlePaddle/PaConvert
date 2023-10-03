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

obj = APIBase("torch.max_pool3d")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.arange(4800, dtype=torch.float32).reshape(2, 3, 8, 10, 10)
        result = torch.max_pool3d(input, 3)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.arange(4800, dtype=torch.float32).reshape(2, 3, 8, 10, 10)
        result = torch.max_pool3d(input, 3, 1, 1, 2, True)
        """
    )
    obj.run(
        pytorch_code, ["result"], unsupport=True, reason="dilation is not suppored now"
    )


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.arange(4800, dtype=torch.float32).reshape(2, 3, 8, 10, 10)
        result = torch.max_pool3d(input=input, kernel_size=3, stride=1, padding=1, ceil_mode=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.arange(4800, dtype=torch.float32).reshape(2, 3, 8, 10, 10)
        result = torch.max_pool3d(input=input, kernel_size=(2, 2, 2), stride=(2, 1, 1), padding=1, ceil_mode=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.arange(4800, dtype=torch.float32).reshape(2, 3, 8, 10, 10)
        result, indices = torch.max_pool3d(input, 2, 2, 1, ceil_mode=True)
        """
    )
    obj.run(
        pytorch_code,
        ["result", "indices"],
        check_dtype=False,
        reason="torch indices dtype is int64, while paddle is int32",
    )
