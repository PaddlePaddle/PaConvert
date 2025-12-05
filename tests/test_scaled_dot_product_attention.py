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

obj = APIBase("torch.nn.functional.scaled_dot_product_attention")


is_cpu = not paddle.device.is_available()


@pytest.mark.skipif(is_cpu, reason="CPU not support fp16/bf16 matmul")
def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import numpy as np
        import torch
        np.random.seed(100)
        x = np.random.rand(32, 8, 128, 64)
        query = torch.tensor(x, dtype=torch.float16)
        result = torch.nn.functional.scaled_dot_product_attention(query, query, query)
        """
    )
    obj.run(pytorch_code, ["result"], atol=1e-3)


@pytest.mark.skipif(is_cpu, reason="CPU not support fp16/bf16 matmul")
def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import numpy as np
        import torch
        np.random.seed(100)
        x = np.random.rand(32, 8, 128, 64)
        query = torch.tensor(x, dtype=torch.float16)
        result = torch.nn.functional.scaled_dot_product_attention(query, query, query, dropout_p=0., is_causal=True)
        """
    )
    obj.run(pytorch_code, ["result"], atol=1e-2)


@pytest.mark.skipif(is_cpu, reason="CPU not support fp16/bf16 matmul")
def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import numpy as np
        import torch
        np.random.seed(100)
        x = np.random.rand(32, 8, 128, 64)
        query = torch.tensor(x, dtype=torch.bfloat16)
        result = torch.nn.functional.scaled_dot_product_attention(query, query, query).float()
        """
    )
    obj.run(pytorch_code, ["result"], atol=1e-2)


@pytest.mark.skipif(is_cpu, reason="CPU not support fp16/bf16 matmul")
def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import numpy as np
        import torch
        np.random.seed(100)
        x = np.random.rand(32, 8, 128, 64)
        query = torch.tensor(x, dtype=torch.bfloat16)
        result = torch.nn.functional.scaled_dot_product_attention(query, query, query, scale=0.2, enable_gqa=False).float()
        """
    )
    obj.run(
        pytorch_code,
    )


@pytest.mark.skipif(is_cpu, reason="CPU not support fp16/bf16 matmul")
def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import numpy as np
        import torch
        np.random.seed(100)
        x = np.random.rand(32, 8, 128, 64)
        mask = np.random.rand(32, 8, 128, 128)
        query = torch.tensor(x, dtype=torch.float16)
        attn_mask = torch.tensor(mask, dtype=torch.float16)
        result = torch.nn.functional.scaled_dot_product_attention(query, query, query, attn_mask=attn_mask)
        """
    )
    obj.run(pytorch_code, ["result"], atol=1e-3)


@pytest.mark.skipif(is_cpu, reason="CPU not support fp16/bf16 matmul")
def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import numpy as np
        import torch
        np.random.seed(100)
        x = np.random.rand(32, 8, 128, 64)
        mask = np.random.rand(32, 8, 128, 128) > 0.5
        query = torch.tensor(x, dtype=torch.float16)
        attn_mask = torch.tensor(mask, dtype=torch.bool)
        result = torch.nn.functional.scaled_dot_product_attention(query, query, query, attn_mask=attn_mask)
        """
    )
    obj.run(pytorch_code, ["result"], atol=1e-3)


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import numpy as np
        import torch
        np.random.seed(100)
        x = np.random.rand(32, 8, 128, 64)
        query = torch.tensor(x, dtype=torch.float64)
        result = torch.nn.functional.scaled_dot_product_attention(query, query, query)
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skipif(is_cpu, reason="CPU not support is_causal=True")
def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import numpy as np
        import torch
        np.random.seed(100)
        x = np.random.rand(32, 8, 128, 64)
        query = torch.tensor(x, dtype=torch.float32)
        result = torch.nn.functional.scaled_dot_product_attention(query, query, query, dropout_p=0., is_causal=True)
        """
    )
    obj.run(pytorch_code, ["result"], atol=1e-5)


def test_case_9():
    pytorch_code = textwrap.dedent(
        """
        import numpy as np
        import torch
        np.random.seed(100)
        x = np.random.rand(32, 8, 128, 64)
        mask = np.random.rand(32, 8, 128, 128)
        query = torch.tensor(x, dtype=torch.float32)
        attn_mask = torch.tensor(mask, dtype=torch.float32)
        result = torch.nn.functional.scaled_dot_product_attention(query, query, query, attn_mask=attn_mask)
        """
    )
    obj.run(pytorch_code, ["result"], atol=1e-5)


@pytest.mark.skipif(is_cpu, reason="CPU not support fp16/bf16 matmul")
def test_case_10():
    pytorch_code = textwrap.dedent(
        """
        import numpy as np
        import torch
        np.random.seed(100)
        # Query: [B, H, S, D], Key/Value: [B, 1, S, D]
        q = np.random.rand(32, 8, 128, 64)
        k = np.random.rand(32, 1, 128, 64)
        v = np.random.rand(32, 1, 128, 64)
        query = torch.tensor(q, dtype=torch.float16)
        key = torch.tensor(k, dtype=torch.float16)
        value = torch.tensor(v, dtype=torch.float16)
        result = torch.nn.functional.scaled_dot_product_attention(query, key, value, enable_gqa=True)
        """
    )
    obj.run(pytorch_code, ["result"], atol=1e-3)


@pytest.mark.skipif(is_cpu, reason="CPU not support fp16/bf16 matmul")
def test_case_11():
    pytorch_code = textwrap.dedent(
        """
        import numpy as np
        import torch
        np.random.seed(100)
        # Query: [B, 8, S, D], Key/Value: [B, 2, S, D]
        q = np.random.rand(32, 8, 128, 64)
        k = np.random.rand(32, 2, 128, 64)
        v = np.random.rand(32, 2, 128, 64)
        query = torch.tensor(q, dtype=torch.float16)
        key = torch.tensor(k, dtype=torch.float16)
        value = torch.tensor(v, dtype=torch.float16)
        result = torch.nn.functional.scaled_dot_product_attention(query, key, value, enable_gqa=True)
        """
    )
    obj.run(pytorch_code, ["result"], atol=1e-3)


def test_case_12():
    pytorch_code = textwrap.dedent(
        """
        import numpy as np
        import torch
        np.random.seed(100)
        # Query len: 64, Key/Val len: 128
        q = np.random.rand(32, 8, 64, 64)
        k = np.random.rand(32, 8, 128, 64)
        v = np.random.rand(32, 8, 128, 64)
        query = torch.tensor(q, dtype=torch.float32)
        key = torch.tensor(k, dtype=torch.float32)
        value = torch.tensor(v, dtype=torch.float32)
        result = torch.nn.functional.scaled_dot_product_attention(query, key, value)
        """
    )
    obj.run(pytorch_code, ["result"], atol=1e-5)


def test_case_13():
    pytorch_code = textwrap.dedent(
        """
        import numpy as np
        import torch
        np.random.seed(100)
        x = np.random.rand(16, 4, 64, 32)
        query = torch.tensor(x, dtype=torch.float64)
        result = torch.nn.functional.scaled_dot_product_attention(query, query, query)
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skipif(is_cpu, reason="CPU not support fp16/bf16 matmul")
def test_case_14():
    pytorch_code = textwrap.dedent(
        """
        import numpy as np
        import torch
        np.random.seed(100)
        x = np.random.rand(32, 8, 128, 64)
        query = torch.tensor(x, dtype=torch.bfloat16)
        result = torch.nn.functional.scaled_dot_product_attention(query, query, query, is_causal=True).float()
        """
    )
    obj.run(pytorch_code, ["result"], atol=1e-2)


def test_case_15():
    pytorch_code = textwrap.dedent(
        """
        import numpy as np
        import torch
        np.random.seed(100)
        x = np.random.rand(32, 8, 64, 64)
        query = torch.tensor(x, dtype=torch.float32)
        # scale set to 0.5
        result = torch.nn.functional.scaled_dot_product_attention(query, query, query, scale=0.5)
        """
    )
    obj.run(pytorch_code, ["result"], atol=1e-5)


@pytest.mark.skipif(is_cpu, reason="CPU not support fp16/bf16 matmul")
def test_case_16():
    pytorch_code = textwrap.dedent(
        """
        import numpy as np
        import torch
        np.random.seed(100)
        x = np.random.rand(32, 8, 64, 64)
        # Mask broadcast over heads
        mask = np.random.rand(32, 1, 64, 64) > 0.5
        query = torch.tensor(x, dtype=torch.float16)
        attn_mask = torch.tensor(mask, dtype=torch.bool)
        result = torch.nn.functional.scaled_dot_product_attention(query, query, query, attn_mask=attn_mask)
        """
    )
    obj.run(pytorch_code, ["result"], atol=1e-3)


def test_case_17():
    pytorch_code = textwrap.dedent(
        """
        import numpy as np
        import torch
        np.random.seed(100)
        x = np.random.rand(16, 4, 32, 32)
        query = torch.tensor(x, dtype=torch.float32)
        # Create additive mask with 0 and -inf
        mask_bool = np.random.rand(16, 4, 32, 32) > 0.5
        mask_float = np.zeros((16, 4, 32, 32))
        mask_float[~mask_bool] = float('-inf')
        attn_mask = torch.tensor(mask_float, dtype=torch.float32)
        result = torch.nn.functional.scaled_dot_product_attention(query, query, query, attn_mask=attn_mask)
        """
    )
    obj.run(pytorch_code, ["result"], atol=1e-5)


def test_case_18():
    pytorch_code = textwrap.dedent(
        """
        import numpy as np
        import torch
        np.random.seed(100)
        x = np.random.rand(32, 8, 64, 64)
        query = torch.tensor(x, dtype=torch.float32)
        result = torch.nn.functional.scaled_dot_product_attention(query, query, query)
        """
    )
    obj.run(pytorch_code, ["result"], atol=1e-5)


@pytest.mark.skipif(is_cpu, reason="CPU not support fp16/bf16 matmul")
def test_case_19():
    pytorch_code = textwrap.dedent(
        """
        import numpy as np
        import torch
        np.random.seed(100)
        q = np.random.rand(32, 8, 128, 64)
        k = np.random.rand(32, 2, 128, 64)
        v = np.random.rand(32, 2, 128, 64)
        query = torch.tensor(q, dtype=torch.float16)
        key = torch.tensor(k, dtype=torch.float16)
        value = torch.tensor(v, dtype=torch.float16)
        result = torch.nn.functional.scaled_dot_product_attention(query, key, value, scale=2.0, enable_gqa=True)
        """
    )
    obj.run(pytorch_code, ["result"], atol=1e-3)


@pytest.mark.skipif(is_cpu, reason="CPU not support fp16/bf16 matmul")
def test_case_20():
    pytorch_code = textwrap.dedent(
        """
        import numpy as np
        import torch
        np.random.seed(100)
        q = np.random.rand(32, 8, 128, 64)
        k = np.random.rand(32, 1, 128, 64)
        v = np.random.rand(32, 1, 128, 64)
        query = torch.tensor(q, dtype=torch.bfloat16)
        key = torch.tensor(k, dtype=torch.bfloat16)
        value = torch.tensor(v, dtype=torch.bfloat16)
        result = torch.nn.functional.scaled_dot_product_attention(query, key, value, is_causal=True, enable_gqa=True).float()
        """
    )
    obj.run(pytorch_code, ["result"], atol=1e-2)


def test_case_21():
    pytorch_code = textwrap.dedent(
        """
        import numpy as np
        import torch
        np.random.seed(100)
        # Shape: [Seq=128, Heads=8, Dim=64] acting as input
        x = np.random.rand(128, 8, 64)
        query = torch.tensor(x, dtype=torch.float32)
        result = torch.nn.functional.scaled_dot_product_attention(query, query, query)
        """
    )
    obj.run(pytorch_code, ["result"], atol=1e-5)


@pytest.mark.skipif(is_cpu, reason="CPU not support fp16/bf16 matmul")
def test_case_22():
    pytorch_code = textwrap.dedent(
        """
        import numpy as np
        import torch
        np.random.seed(100)
        # Q: [B, H, 32, D], K: [B, H, 64, D]
        q = np.random.rand(16, 4, 32, 32)
        k = np.random.rand(16, 4, 64, 32)
        v = np.random.rand(16, 4, 64, 32)
        # Mask shape must match broadcast shape: [B, H, 32, 64]
        mask = np.random.rand(16, 4, 32, 64) > 0.5

        query = torch.tensor(q, dtype=torch.float16)
        key = torch.tensor(k, dtype=torch.float16)
        value = torch.tensor(v, dtype=torch.float16)
        attn_mask = torch.tensor(mask, dtype=torch.bool)
        result = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask)
        """
    )
    obj.run(pytorch_code, ["result"], atol=1e-3)


def test_case_23():
    pytorch_code = textwrap.dedent(
        """
        import numpy as np
        import torch
        np.random.seed(100)
        x = np.random.rand(16, 4, 32, 32)
        mask = np.random.rand(16, 4, 32, 32) > 0.5
        query = torch.tensor(x, dtype=torch.float32)
        attn_mask = torch.tensor(mask, dtype=torch.bool)
        result = torch.nn.functional.scaled_dot_product_attention(query, query, query, attn_mask=attn_mask, is_causal=False)
        """
    )
    obj.run(pytorch_code, ["result"], atol=1e-5)


@pytest.mark.skipif(is_cpu, reason="CPU not support fp16/bf16 matmul")
def test_case_24():
    pytorch_code = textwrap.dedent(
        """
        import numpy as np
        import torch
        np.random.seed(100)
        x = np.random.rand(32, 8, 128, 64)
        query = torch.tensor(x, dtype=torch.float16)
        # scale=None should default to 1/sqrt(head_dim)
        result = torch.nn.functional.scaled_dot_product_attention(query, query, query, dropout_p=0.0, scale=None)
        """
    )
    obj.run(pytorch_code, ["result"], atol=1e-3)


@pytest.mark.skipif(is_cpu, reason="CPU not support fp16/bf16 matmul")
def test_case_25():
    pytorch_code = textwrap.dedent(
        """
        import numpy as np
        import torch
        np.random.seed(100)
        x = np.random.rand(32, 8, 128, 64)
        query = torch.tensor(x, dtype=torch.float16)
        result = torch.nn.functional.scaled_dot_product_attention(query, query, query, enable_gqa=True)
        """
    )
    obj.run(pytorch_code, ["result"], atol=1e-3)


def test_case_26():
    pytorch_code = textwrap.dedent(
        """
        import numpy as np
        import torch
        np.random.seed(100)
        # Smallest dimensions: Batch=1, Head=1, Seq=1, Dim=1
        x = np.random.rand(1, 1, 1, 1)
        query = torch.tensor(x, dtype=torch.float32)
        result = torch.nn.functional.scaled_dot_product_attention(query, query, query)
        """
    )
    obj.run(pytorch_code, ["result"], atol=1e-5)


@pytest.mark.skipif(is_cpu, reason="CPU not support fp16/bf16 matmul")
def test_case_27():
    pytorch_code = textwrap.dedent(
        """
        import numpy as np
        import torch
        np.random.seed(100)
        x = np.random.rand(32, 8, 64, 64)
        query = torch.tensor(x, dtype=torch.bfloat16)
        result = torch.nn.functional.scaled_dot_product_attention(query, query, query, scale=0.1).float()
        """
    )
    obj.run(pytorch_code, ["result"], atol=1e-2)


def test_case_28():
    pytorch_code = textwrap.dedent(
        """
        import numpy as np
        import torch
        np.random.seed(100)
        q = np.random.rand(10, 4, 32, 16)
        mask = np.random.rand(1, 4, 32, 32) > 0.5
        query = torch.tensor(q, dtype=torch.float32)
        attn_mask = torch.tensor(mask, dtype=torch.bool)
        result = torch.nn.functional.scaled_dot_product_attention(query, query, query, attn_mask=attn_mask)
        """
    )
    obj.run(pytorch_code, ["result"], atol=1e-5)


@pytest.mark.skipif(is_cpu, reason="CPU not support fp16/bf16 matmul")
def test_case_29():
    pytorch_code = textwrap.dedent(
        """
        import numpy as np
        import torch
        np.random.seed(100)
        # GQA: Q heads=8, K/V heads=4
        q = np.random.rand(32, 8, 128, 64)
        k = np.random.rand(32, 4, 128, 64)
        v = np.random.rand(32, 4, 128, 64)
        query = torch.tensor(q, dtype=torch.float16)
        key = torch.tensor(k, dtype=torch.float16)
        value = torch.tensor(v, dtype=torch.float16)
        result = torch.nn.functional.scaled_dot_product_attention(query, key, value, is_causal=True, enable_gqa=True)
        """
    )
    obj.run(pytorch_code, ["result"], atol=1e-3)
