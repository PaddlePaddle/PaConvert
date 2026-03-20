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

obj = APIBase("torch.Tensor.nanquantile")


def test_case_1():
    """Basic usage with positional arguments"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, 2.0, float('nan')], [4.0, 5.0, 6.0]])
        result = x.nanquantile(0.5)
    """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    """With dim parameter as positional argument"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, 2.0, float('nan')], [4.0, 5.0, 6.0]])
        result = x.nanquantile(0.5, 1)
    """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    """With dim and keepdim parameters as positional arguments"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, 2.0, float('nan')], [4.0, 5.0, 6.0]])
        result = x.nanquantile(0.5, 0, True)
    """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    """With dim keyword argument (using alias)"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, 2.0, float('nan')], [4.0, 5.0, 6.0]])
        result = x.nanquantile(0.5, dim=1)
    """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    """With keepdim keyword argument"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, 2.0, float('nan')], [4.0, 5.0, 6.0]])
        result = x.nanquantile(0.25, 0, keepdim=True)
    """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    """Multiple quantile values"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, 2.0, float('nan')], [4.0, 5.0, 6.0]])
        result = x.nanquantile(torch.tensor([0.25, 0.5, 0.75]))
    """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    """Keywords in different order (dim first)"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, 2.0, float('nan')], [4.0, 5.0, 6.0]])
        result = x.nanquantile(q=0.5, dim=1, keepdim=False)
    """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    """Keywords completely out of order"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, 2.0, float('nan')], [4.0, 5.0, 6.0]])
        result = x.nanquantile(dim=1, q=0.5, keepdim=True)
    """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    """1D tensor input"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1.0, 2.0, float('nan'), 4.0, 5.0])
        result = x.nanquantile(0.5)
    """
    )
    obj.run(pytorch_code, ["result"])


def test_case_10():
    """3D tensor input"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[[1.0, float('nan')], [3.0, 4.0]], [[5.0, 6.0], [float('nan'), 8.0]]])
        result = x.nanquantile(0.5, 2)
    """
    )
    obj.run(pytorch_code, ["result"])


def test_case_11():
    """Mixed parameter styles"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, 2.0, float('nan')], [4.0, 5.0, 6.0]])
        result = x.nanquantile(0.5, dim=1, keepdim=True)
    """
    )
    obj.run(pytorch_code, ["result"])


def test_case_12():
    """Verify NaN handling - all NaN row"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[float('nan'), float('nan'), float('nan')], [1.0, 2.0, 3.0]])
        result = x.nanquantile(0.5, 1)
    """
    )
    obj.run(pytorch_code, ["result"])


def test_case_13():
    """Quantile at boundaries"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, 2.0, float('nan')], [4.0, 5.0, 6.0]])
        result = x.nanquantile(0.0, 1)
    """
    )
    obj.run(pytorch_code, ["result"])


def test_case_14():
    """Quantile at upper boundary"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, 2.0, float('nan')], [4.0, 5.0, 6.0]])
        result = x.nanquantile(1.0, 1)
    """
    )
    obj.run(pytorch_code, ["result"])


def test_case_15():
    """Chained method calls"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, 2.0, float('nan')], [4.0, 5.0, 6.0]])
        result = x.nanquantile(0.5, 1).nanquantile(0.5)
    """
    )
    obj.run(pytorch_code, ["result"])
