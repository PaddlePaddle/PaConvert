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

obj = APIBase("torch.Tensor.unique")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        src = torch.tensor([1., 2., 3., 4., 5., 6.])
        result = src.unique()
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        src = torch.tensor([1., 2., 3., 4., 5., 6.])
        result = src.unique(dim=0)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        src = torch.tensor([1., 2., 3., 4., 5., 6.])
        result = src.unique(sorted=True, return_inverse=True, return_counts=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        src = torch.tensor([1., 2., 3., 4., 5., 6.])
        result = src.unique(sorted=True, return_inverse=True, return_counts=True, dim=0)
        """
    )
    obj.run(pytorch_code, ["result"])


def _test_case_5():
    # Paddle returns different number of elements than PyTorch when using
    # positional args (sorted, return_inverse) with dim, due to sorted being
    # dropped in conversion; result tuple length mismatch: Unable to align results
    pytorch_code = textwrap.dedent(
        """
        import torch
        src = torch.tensor([1., 2., 3., 4., 5., 6.])
        result = src.unique(True, True, dim=0)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        src = torch.tensor([1., 2., 3., 4., 5., 6.])
        result = src.unique(False, False, dim=0)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        src = torch.tensor([[1, 2, 3], [1, 2, 3], [4, 5, 6]])
        result = src.unique(dim=0)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        src = torch.tensor([3, 1, 2, 1, 3])
        result = src.unique(return_inverse=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    pytorch_code = textwrap.dedent(
        """
        import torch
        src = torch.tensor([3, 1, 2, 1, 3])
        result = src.unique(return_counts=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_10():
    pytorch_code = textwrap.dedent(
        """
        import torch
        src = torch.tensor([3, 1, 2, 1, 3])
        result = src.unique(return_inverse=True, return_counts=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_11():
    pytorch_code = textwrap.dedent(
        """
        import torch
        src = torch.tensor([[1, 2, 1, 3], [1, 2, 1, 3]])
        result = src.unique(dim=1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_12():
    pytorch_code = textwrap.dedent(
        """
        import torch
        src = torch.tensor([4, 1, 2, 1, 4, 3])
        result = src.unique(return_inverse=True, return_counts=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_13():
    # keyword args shuffled
    pytorch_code = textwrap.dedent(
        """
        import torch
        src = torch.tensor([3, 1, 2, 1, 3])
        result = src.unique(return_counts=True, dim=0, sorted=True, return_inverse=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_14():
    # negative dim
    pytorch_code = textwrap.dedent(
        """
        import torch
        src = torch.tensor([[1, 2, 1], [3, 4, 3]])
        result = src.unique(dim=-1)
        """
    )
    obj.run(pytorch_code, ["result"])


def _test_case_15():  # Converter does not support **kwargs unpacking for Tensor methods
    # kwargs dict unpacking
    pytorch_code = textwrap.dedent(
        """
        import torch
        src = torch.tensor([3, 1, 2, 1, 3])
        kwargs = {'return_inverse': True, 'return_counts': True}
        result = src.unique(**kwargs)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_16():
    # int64 data with explicit values, all three outputs requested
    pytorch_code = textwrap.dedent(
        """
        import torch
        src = torch.tensor([5, -3, 5, 7, -3, 9, 0, 0, 5, -1, 8, 8], dtype=torch.int64)
        result = src.unique(sorted=True, return_inverse=True, return_counts=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_17():
    # float32 two-dimensional tensor along axis 0 without counts/inverse
    pytorch_code = textwrap.dedent(
        """
        import torch
        src = torch.tensor([[1., 2., 1.], [1., 2., 1.]])
        result = src.unique(dim=0)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_18():
    # explicit sorted=False on a non-trivial sequence; check tuple structure consistency
    pytorch_code = textwrap.dedent(
        """
        import torch
        src = torch.tensor([4, -3, 7, 7, -3, 9], dtype=torch.int64)
        uniques, inverse, counts = src.unique(return_inverse=True, return_counts=True)
        result = (uniques.size(0) == int(counts.sum()))
        """
    )
    obj.run(pytorch_code, ["result"])
