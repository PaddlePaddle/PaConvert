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

obj = APIBase("torch.testing.make_tensor")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.testing.make_tensor((3,), device='cpu', dtype=torch.float32)
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        unsupport=True,
        reason="paddle does not support this function temporarily",
    )


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.testing.make_tensor((3,), dtype=torch.float32, device='cpu')
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        unsupport=True,
        reason="paddle does not support this function temporarily",
    )


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.testing.make_tensor((2, 2), device='cuda', dtype=torch.bool)
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        unsupport=True,
        reason="paddle does not support this function temporarily",
    )


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.testing.make_tensor((3,), dtype=torch.float32, device='cpu',
                                           low=None, high=None, requires_grad=False, noncontiguous=False, exclude_zero=False,
                                           memory_format=None)
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        unsupport=True,
        reason="paddle does not support this function temporarily",
    )


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.testing.make_tensor((3,), dtype=torch.float32, device='cpu',
                                           low=-1, high=1, requires_grad=False, noncontiguous=False, exclude_zero=False,
                                           memory_format=None)
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        unsupport=True,
        reason="paddle does not support this function temporarily",
    )
