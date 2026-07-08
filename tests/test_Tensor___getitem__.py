# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

obj = APIBase("torch.Tensor.__getitem__")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = x[1, 2]
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = x[:, 1:3]
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1, 2, 3, 4, 5])
        indices = torch.tensor([0, 2, 4])
        result = x[indices]
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    """Multi-dimensional int tensor indexing"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        idx0 = torch.tensor([0, 1, 2])
        idx1 = torch.tensor([2, 1, 0])
        result = x[idx0, idx1]
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    """2D broadcast int tensor indexing on 3D tensor"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.arange(24).reshape(2, 3, 4).float()
        idx0 = torch.tensor([[0, 1], [1, 1]])
        idx1 = torch.tensor([[0, 2], [1, 0]])
        result = x[idx0, idx1]
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    """Bool tensor indexing (1D bool on 2D tensor)"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        mask = torch.tensor([True, False, True])
        result = x[mask]
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    """Bool tensor indexing with 2D mask matching input shape"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        mask = torch.tensor([[True, False, True], [False, True, False], [True, False, True]])
        result = x[mask]
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    """Negative indices in tensor"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        idx0 = torch.tensor([0, -1, 2])
        idx1 = torch.tensor([-1, 0, -2])
        result = x[idx0, idx1]
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    """Trailing dims: fewer indices than dimensions"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.arange(24).reshape(2, 3, 4).float()
        idx = torch.tensor([0, 1])
        result = x[idx]
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_10():
    """Trailing dims: 1D index on 3D tensor"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.arange(120).reshape(4, 5, 6).float()
        idx0 = torch.tensor([0, 2])
        idx1 = torch.tensor([1, 3])
        result = x[idx0, idx1]
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_11():
    """Gradient computation with tensor indexing"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1., 2., 3.], [4., 5., 6.]], requires_grad=True)
        idx0 = torch.tensor([0, 1])
        idx1 = torch.tensor([2, 0])
        y = x[idx0, idx1]
        y.sum().backward()
        x_grad = x.grad
        """
    )
    obj.run(pytorch_code, ["y", "x_grad"], check_stop_gradient=False)


def test_case_12():
    """Mixed indexing: tensor for first dim, slice for second"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        idx = torch.tensor([0, 2])
        result = x[idx, 0:2]
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_13():
    """Float32 dtype indexing"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.arange(12).reshape(3, 4).float()
        idx0 = torch.tensor([0, 2])
        idx1 = torch.tensor([1, 3])
        result = x[idx0, idx1]
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_14():
    """Float64 dtype indexing"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.arange(12).reshape(3, 4).double()
        idx0 = torch.tensor([0, 1])
        idx1 = torch.tensor([2, 3])
        result = x[idx0, idx1]
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_15():
    """Int64 index dtype (torch.long)"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[10, 20, 30], [40, 50, 60]])
        idx0 = torch.tensor([0, 1], dtype=torch.long)
        idx1 = torch.tensor([2, 0], dtype=torch.long)
        result = x[idx0, idx1]
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_16():
    """Single element result"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 2, 3], [4, 5, 6]])
        result = x[1, 2]
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_17():
    """1D tensor with 1D tensor indexing"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.arange(10).float()
        idx = torch.tensor([9, 0, 5, 3])
        result = x[idx]
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_18():
    """202 broadcast int tensor indexing (2D indices)"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.arange(60).reshape(5, 4, 3).float()
        idx0 = torch.tensor([[0, 1], [2, 3]])
        idx1 = torch.tensor([[0, 2], [1, 1]])
        result = x[idx0, idx1]
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_19():
    """All-false bool mask (empty result)"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 2], [3, 4]])
        mask = torch.tensor([False, False])
        result = x[mask]
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_20():
    """Bool tensor indexing on 3D tensor"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.arange(24).reshape(2, 3, 4).float()
        mask = torch.tensor([True, False])
        result = x[mask]
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_21():
    """Broadcast indices with 1-rank difference"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.arange(100).reshape(10, 10).float()
        idx0 = torch.tensor([1, 2, 3]).reshape(3, 1)
        idx1 = torch.tensor([1, 2, 3, 4]).reshape(1, 4)
        result = x[idx0, idx1]
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_22():
    """Large tensor indexing"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.arange(1000).reshape(10, 100).float()
        idx = torch.arange(0, 100, 10)
        result = x[:, idx]
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_23():
    """Int32 index dtype"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
        idx0 = torch.tensor([0, 2], dtype=torch.int32)
        idx1 = torch.tensor([1, 0], dtype=torch.int32)
        result = x[idx0, idx1]
        """
    )
    obj.run(pytorch_code, ["result"])
