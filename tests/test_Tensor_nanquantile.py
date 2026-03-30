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
    """With keepdim keyword argument"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, 2.0, float('nan')], [4.0, 5.0, 6.0]])
        result = x.nanquantile(0.25, 0, keepdim=True)
    """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    """Multiple quantile values"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, 2.0, float('nan')], [4.0, 5.0, 6.0]])
        result = x.nanquantile(torch.tensor([0.25, 0.5, 0.75]))
    """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    """3D tensor input"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[[1.0, float('nan')], [3.0, 4.0]], [[5.0, 6.0], [float('nan'), 8.0]]])
        result = x.nanquantile(0.5, 2)
    """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    """Mixed parameter styles with dim and keepdim"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, 2.0, float('nan')], [4.0, 5.0, 6.0]])
        result = x.nanquantile(0.5, dim=1, keepdim=True)
    """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    """Verify NaN handling - all NaN row"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[float('nan'), float('nan'), float('nan')], [1.0, 2.0, 3.0]])
        result = x.nanquantile(0.5, 1)
    """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    """Quantile at lower boundary"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, 2.0, float('nan')], [4.0, 5.0, 6.0]])
        result = x.nanquantile(0.0, 1)
    """
    )
    obj.run(pytorch_code, ["result"])


def test_case_10():
    """Quantile at upper boundary"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, 2.0, float('nan')], [4.0, 5.0, 6.0]])
        result = x.nanquantile(1.0, 1)
    """
    )
    obj.run(pytorch_code, ["result"])


def test_case_11():
    """Chained method calls"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, 2.0, float('nan')], [4.0, 5.0, 6.0]])
        result = x.nanquantile(0.5, 1).nanquantile(0.5)
    """
    )
    obj.run(pytorch_code, ["result"])


def test_case_12():
    """With interpolation='lower'"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, 2.0, float('nan')], [4.0, 5.0, 6.0]])
        result = x.nanquantile(0.5, dim=1, interpolation='lower')
    """
    )
    obj.run(pytorch_code, ["result"])


def test_case_13():
    """With interpolation='higher'"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, 2.0, float('nan')], [4.0, 5.0, 6.0]])
        result = x.nanquantile(0.5, dim=1, interpolation='higher')
    """
    )
    obj.run(pytorch_code, ["result"])


def test_case_14():
    """With interpolation='midpoint'"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, 2.0, float('nan')], [4.0, 5.0, 6.0]])
        result = x.nanquantile(0.5, dim=1, interpolation='midpoint')
    """
    )
    obj.run(pytorch_code, ["result"])


def test_case_15():
    """With interpolation='nearest'"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, 2.0, float('nan')], [4.0, 5.0, 6.0]])
        result = x.nanquantile(0.5, dim=1, interpolation='nearest')
    """
    )
    obj.run(pytorch_code, ["result"])


def test_case_16():
    """Multiple quantiles with interpolation parameter"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, 2.0, float('nan')], [4.0, 5.0, 6.0]])
        result = x.nanquantile(torch.tensor([0.25, 0.5, 0.75]), dim=1, interpolation='midpoint')
    """
    )
    obj.run(pytorch_code, ["result"])


def test_case_17():
    """With dim=None explicitly (flatten behavior)"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, 2.0, float('nan')], [4.0, 5.0, 6.0]])
        result = x.nanquantile(0.5, dim=None)
    """
    )
    obj.run(pytorch_code, ["result"])


def test_case_18():
    """With dim=None and keepdim=True"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, 2.0, float('nan')], [4.0, 5.0, 6.0]])
        result = x.nanquantile(0.5, dim=None, keepdim=True)
    """
    )
    obj.run(pytorch_code, ["result"])


def test_case_19():
    """Multiple quantiles with interpolation on 3D tensor"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[[1.0, float('nan')], [3.0, 4.0]], [[5.0, 6.0], [float('nan'), 8.0]]])
        result = x.nanquantile(torch.tensor([0.25, 0.75]), dim=0, interpolation='lower')
    """
    )
    obj.run(pytorch_code, ["result"])


def test_case_20():
    """All parameters including keywords in mixed order"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, 2.0, float('nan')], [4.0, 5.0, 6.0]])
        result = x.nanquantile(q=0.5, interpolation='higher', dim=1, keepdim=False)
    """
    )
    obj.run(pytorch_code, ["result"])
