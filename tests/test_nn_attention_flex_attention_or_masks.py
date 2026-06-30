# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

obj = APIBase("torch.nn.attention.flex_attention.or_masks")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torch.nn.attention.flex_attention import or_masks

        def mask_a(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        def mask_b(b, h, q_idx, kv_idx):
            return h == 0

        mask = or_masks(mask_a, mask_b)
        result = mask(
            torch.tensor([0]),
            torch.tensor([1]),
            torch.tensor([2]),
            torch.tensor([3]),
        )
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torch.nn.attention.flex_attention import or_masks

        def mask_a(b, h, q_idx, kv_idx):
            return q_idx <= kv_idx

        args = (mask_a,)
        mask = or_masks(*args)
        result = mask(
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([1]),
            torch.tensor([1]),
        )
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        from torch.nn.attention.flex_attention import or_masks

        mask = or_masks()
        result = mask(
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([1]),
            torch.tensor([2]),
        )
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import pytest
        from torch.nn.attention.flex_attention import or_masks

        with pytest.raises(RuntimeError):
            or_masks(lambda b, h, q_idx, kv_idx: q_idx >= kv_idx, 1)
        """
    )
    obj.run(pytorch_code, [])
