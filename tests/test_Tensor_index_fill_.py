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

obj = APIBase("torch.Tensor.index_fill_")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.eye(2, 4)
        indices = torch.tensor([0, 1])
        value = -1
        result = x.index_fill_(0, indices, value)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.eye(3, 4)
        indices = torch.tensor([0, 1])
        value = -1
        result = x.index_fill_(1, indices, value)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.eye(3, 4)
        indices = torch.tensor([0, 1])
        dim = 0
        value = -1
        result = x.index_fill_(index=indices, dim=dim, value=value)
        """
    )
    obj.run(pytorch_code, ["result", "x"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.eye(3, 4)
        indices = torch.tensor([0, 1])
        dim = 0
        value = -1
        result = x.index_fill_(dim=dim, index=indices, value=value)
        """
    )
    obj.run(pytorch_code, ["result", "x"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.eye(3, 4)
        indices = torch.tensor([0, 3])
        value = -1
        result = x.index_fill_(1, indices, value)
        """
    )
    obj.run(pytorch_code, ["result", "x"])


def test_case_6():
    """Mixed positional and keyword arguments test"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.eye(3, 4)
        indices = torch.tensor([0, 1])
        result = x.index_fill_(0, index=indices, value=-1)
        """
    )
    obj.run(pytorch_code, ["result", "x"])


def test_case_7():
    """Expression argument test for dim"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.eye(3, 4)
        indices = torch.tensor([0, 1])
        result = x.index_fill_(1 + 0, indices, -1)
        """
    )
    obj.run(pytorch_code, ["result", "x"])


def test_case_8():
    """Float64 dtype test"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.eye(3, 4, dtype=torch.float64)
        indices = torch.tensor([0, 1])
        value = -2.5
        result = x.index_fill_(0, indices, value)
        """
    )
    obj.run(pytorch_code, ["result", "x"])


def test_case_9():
    """3D tensor test"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.ones(2, 3, 4)
        indices = torch.tensor([0, 2])
        value = -1
        result = x.index_fill_(1, indices, value)
        """
    )
    obj.run(pytorch_code, ["result", "x"])


def test_case_10():
    """4D tensor test"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.ones(2, 3, 4, 5)
        indices = torch.tensor([0, 1, 3])
        value = -1
        result = x.index_fill_(2, indices, value)
        """
    )
    obj.run(pytorch_code, ["result", "x"])


def test_case_11():
    """Tensor value parameter test"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.eye(3, 4)
        indices = torch.tensor([0, 1])
        value = torch.tensor(-1.0)
        result = x.index_fill_(0, indices, value)
        """
    )
    obj.run(pytorch_code, ["result", "x"])


def test_case_12():
    """Int32 dtype test"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.ones(3, 4, dtype=torch.int32)
        indices = torch.tensor([0, 1])
        value = -1
        result = x.index_fill_(0, indices, value)
        """
    )
    obj.run(pytorch_code, ["result", "x"])


def test_case_13():
    """Out of order keyword arguments with all parameters"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.eye(3, 4)
        indices = torch.tensor([0, 1])
        result = x.index_fill_(value=-1, index=indices, dim=0)
        """
    )
    obj.run(pytorch_code, ["result", "x"])
