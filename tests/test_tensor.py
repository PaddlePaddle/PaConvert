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

import paddle
import pytest
from apibase import APIBase

obj = APIBase("torch.tensor")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.tensor([2, 3])
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        data = [2, 3]
        result = torch.tensor(data)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        data = [2, 3]
        result = torch.tensor(data, dtype=torch.float)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        data = [2, 3]
        result = torch.tensor(data, dtype=torch.float, device=None)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        data = [2, 3]
        result = torch.tensor(data, dtype=torch.float, device=None, requires_grad = False)
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="can only run on paddle with CUDA",
)
def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        data = [2, 3]
        result = torch.tensor(data, requires_grad=False, pin_memory=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        data = [2, 3]
        result = torch.tensor(data, requires_grad = False, pin_memory=False)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        data = [2, 3]
        result = torch.tensor(data=data, dtype=torch.float, device=None, requires_grad=False, pin_memory=False)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    pytorch_code = textwrap.dedent(
        """
        import torch
        data = [2, 3]
        result = torch.tensor(device=None, dtype=torch.float, pin_memory=False, data=data, requires_grad=False)
        """
    )
    obj.run(pytorch_code, ["result"])
