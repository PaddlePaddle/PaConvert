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

obj = APIBase("torch.Tensor.cummin")


def test_case_1():
    """Test basic usage with positional and keyword arguments"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = x.cummin(0)
        result2 = x.cummin(dim=1)
        """
    )
    obj.run(pytorch_code, ["result", "result2"])


def test_case_2():
    """Test with 1D and 3D tensors"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x1d = torch.tensor([9.0, 5.0, 2.0, 7.0])
        result1 = x1d.cummin(0)

        x3d = torch.tensor([[[12.0, 11.0], [10.0, 9.0]], [[8.0, 7.0], [6.0, 5.0]]])
        result2 = x3d.cummin(dim=1)
        result3 = x3d.cummin(dim=2)
        """
    )
    obj.run(pytorch_code, ["result1", "result2", "result3"])


def test_case_3():
    """Test with different dtypes and values"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x_float64 = torch.tensor([[6.0, 5.0]], dtype=torch.float64)
        result1 = x_float64.cummin(0)

        x_neg = torch.tensor([[-1.0, 2.0], [-3.0, 4.0]])
        result2 = x_neg.cummin(dim=1)
        """
    )
    obj.run(pytorch_code, ["result1", "result2"])


def test_case_4():
    """Test NamedTuple access (values, indices and [0], [1])"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[4.0, 2.0], [1.0, 3.0]])
        result = x.cummin(0)
        values = result.values
        indices = result.indices
        v0 = result[0]
        i1 = result[1]
        """
    )
    obj.run(pytorch_code, ["values", "indices", "v0", "i1"])


def test_case_5():
    """Test NamedTuple access with 3D tensor"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[[12.0, 11.0], [10.0, 9.0]], [[8.0, 7.0], [6.0, 5.0]]])
        result = x.cummin(dim=1)
        values = result.values
        indices = result.indices
        """
    )
    obj.run(pytorch_code, ["values", "indices"])


def test_case_6():
    """Test NamedTuple access with negative values"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[-1.0, 2.0], [-3.0, 4.0]])
        result = x.cummin(dim=1)
        values = result.values
        indices = result.indices
        """
    )
    obj.run(pytorch_code, ["values", "indices"])
