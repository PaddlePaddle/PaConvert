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

obj = APIBase("torch.cummax")


def test_case_1():
    """Test basic usage with positional and keyword arguments"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = torch.cummax(x, 0)
        result2 = torch.cummax(x, dim=1)
        result3 = torch.cummax(input=x, dim=0)
        """
    )
    obj.run(pytorch_code, ["result", "result2", "result3"])


def test_case_2():
    """Test with out parameter (tuple)"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        values = torch.empty(2, 2)
        indices = torch.empty(2, 2, dtype=torch.int64)
        result = torch.cummax(x, 0, out=(values, indices))
        """
    )
    obj.run(pytorch_code, ["result", "values", "indices"])


def test_case_3():
    """Test with 1D and 3D tensors"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x1d = torch.tensor([3.0, 1.0, 4.0, 1.0, 5.0])
        result1 = torch.cummax(x1d, 0)

        x3d = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        result2 = torch.cummax(x3d, dim=1)
        result3 = torch.cummax(x3d, dim=2)
        """
    )
    obj.run(pytorch_code, ["result1", "result2", "result3"])


def test_case_4():
    """Test with different dtypes and values"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x_float64 = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
        result1 = torch.cummax(x_float64, 0)

        x_neg = torch.tensor([[-1.0, 2.0], [-3.0, 4.0]])
        result2 = torch.cummax(x_neg, dim=1)
        """
    )
    obj.run(pytorch_code, ["result1", "result2"])


def test_case_5():
    """Test NamedTuple access (values, indices and [0], [1])"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = torch.cummax(x, 0)
        values = result.values
        indices = result.indices
        v0 = result[0]
        i1 = result[1]
        """
    )
    obj.run(pytorch_code, ["values", "indices", "v0", "i1"])


def test_case_6():
    """Test with out parameter (list)"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        values = torch.empty(2, 2)
        indices = torch.empty(2, 2, dtype=torch.int64)
        result = torch.cummax(x, 0, out=[values, indices])
        """
    )
    obj.run(pytorch_code, ["result", "values", "indices"])


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        values = torch.empty(2, 2)
        indices = torch.empty(2, 2, dtype=torch.int64)
        out = (values, indices)
        result = torch.cummax(out=out, dim=0, input=x)
        """
    )
    obj.run(pytorch_code, ["result", "out"])
