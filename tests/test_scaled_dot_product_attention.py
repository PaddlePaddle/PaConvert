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


# can not run by flash attention backend
@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="cpu matmul not supoort float16",
)
def _test_case_1():
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
    obj.run(pytorch_code, ["result"], rtol=1e-3)


# can not run by flash attention backend
@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="cpu matmul not supoort float16",
)
def _test_case_2():
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
    obj.run(pytorch_code, ["result"], rtol=1e-3)


# can not run by flash attention backend
@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="cpu matmul not supoort bfloat16",
)
def _test_case_3():
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
    obj.run(pytorch_code, ["result"], rtol=1e-2)


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
        unsupport=True,
        reason="paddle not support 'scale' and 'enable_gqa' ",
    )


# can not run by flash attention backend
@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="cpu matmul not supoort float16",
)
def _test_case_5():
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
    obj.run(pytorch_code, ["result"], rtol=1e-3)


# can not run by flash attention backend
@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="cpu matmul not supoort float16",
)
def _test_case_6():
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
    obj.run(pytorch_code, ["result"], rtol=1e-3)


# can not run by flash attention backend
def _test_case_7():
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


# can not run by flash attention backend
def _test_case_8():
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
    obj.run(pytorch_code, ["result"], rtol=1e-5)


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
    obj.run(pytorch_code, ["result"], rtol=1e-5)


def test_case_10():
    pytorch_code = textwrap.dedent(
        """
        import numpy as np
        import torch
        np.random.seed(100)
        x = np.random.rand(8, 128, 64)
        query = torch.tensor(x, dtype=torch.float16)
        result = torch.nn.functional.scaled_dot_product_attention(query, query, query)
        """
    )
    obj.run(pytorch_code, ["result"], rtol=1e-5)
