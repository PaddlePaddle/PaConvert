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

obj = APIBase("torch.Tensor.logical_or")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.tensor([True, False, True]).logical_or(torch.tensor([True, False, False]))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([0, 1, 10, 0], dtype=torch.int8)
        b = torch.tensor([4, 0, 1, 0], dtype=torch.int8)
        result = a.logical_or(b)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([0, 1, 10, 0], dtype=torch.float32)
        b = torch.tensor([4, 0, 1, 0], dtype=torch.float32)
        result = a.logical_or(b)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([0, 1, 10, 0], dtype=torch.float32)
        b = torch.tensor([4, 0, 1, 0], dtype=torch.float32)
        result = torch.tensor([0, 1, 10., 0.]).logical_or(other=torch.tensor([4, 0, 10., 0.]))
        """
    )
    obj.run(pytorch_code, ["result"])


# paddle not support type promote
def _test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([0, 1, 10, 0], dtype=torch.float32)
        b = torch.tensor([4, 0, 1, 0], dtype=torch.int8)
        result = a.logical_or(b)
        """
    )
    obj.run(pytorch_code, ["result"])
