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

obj = APIBase("torch.nn.MultiheadAttention")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        import numpy as np
        np.random.seed(42)
        query = torch.ones([4, 32, 128])
        key = torch.ones([4, 32, 128])
        value = torch.ones([4, 32, 128])
        multihead_attn = nn.MultiheadAttention(128, 2, 0.0, True)
        for param in multihead_attn.parameters():
            param.data = torch.from_numpy(np.random.random(param.shape).astype('float32'))
            param.requires_grad = True
        multihead_attn.eval()
        result = multihead_attn(query, key, value)
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
    )


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        import numpy as np
        np.random.seed(42)
        query = torch.ones([4, 32, 128])
        key = torch.ones([4, 32, 128])
        value = torch.ones([4, 32, 128])
        multihead_attn = nn.MultiheadAttention(embed_dim=128, num_heads=2, dropout=0.0, bias=True)
        for param in multihead_attn.parameters():
            param.data = torch.from_numpy(np.random.random(param.shape).astype('float32'))
            param.requires_grad = True
        multihead_attn.eval()
        result = multihead_attn(query, key, value)
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
    )


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        import numpy as np
        np.random.seed(42)
        query = torch.ones([4, 32, 128])
        key = torch.ones([4, 32, 128])
        value = torch.ones([4, 32, 128])
        multihead_attn = nn.MultiheadAttention(embed_dim=128, num_heads=2, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False)
        for param in multihead_attn.parameters():
            param.data = torch.from_numpy(np.random.random(param.shape).astype('float32'))
            param.requires_grad = True
        multihead_attn.eval()
        result = multihead_attn(query, key, value)
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
    )


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        import numpy as np
        np.random.seed(42)
        query = torch.ones([4, 32, 128])
        key = torch.ones([4, 32, 128])
        value = torch.ones([4, 32, 128])
        multihead_attn = nn.MultiheadAttention(128, 2, 0.5, False, kdim=128, vdim=128)
        for param in multihead_attn.parameters():
            param.data = torch.from_numpy(np.random.random(param.shape).astype('float32'))
            param.requires_grad = True
        multihead_attn.eval()
        result = multihead_attn(query, key, value)
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
    )


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        import numpy as np
        np.random.seed(42)
        query = torch.ones([4, 32, 128])
        key = torch.ones([4, 32, 128])
        value = torch.ones([4, 32, 128])
        multihead_attn = nn.MultiheadAttention(128, 2, 0.5, False, batch_first=True)
        for param in multihead_attn.parameters():
            param.data = torch.from_numpy(np.random.random(param.shape).astype('float32'))
            param.requires_grad = True
        multihead_attn.eval()
        result = multihead_attn(query, key, value)
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
    )


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        import numpy as np
        np.random.seed(42)
        query = torch.ones([4, 32, 128])
        key = torch.ones([4, 32, 128])
        value = torch.ones([4, 32, 128])
        multihead_attn = nn.MultiheadAttention(128, 2, kdim=128, vdim=128, device=torch.device('cpu'), dtype=torch.float32)
        for param in multihead_attn.parameters():
            param.data = torch.from_numpy(np.random.random(param.shape).astype('float32'))
            param.requires_grad = True
        multihead_attn.eval()
        result = multihead_attn(query, key, value)
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
    )


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        import numpy as np
        np.random.seed(42)
        query = torch.ones([4, 32, 128])
        key = torch.ones([4, 32, 128])
        value = torch.ones([4, 32, 128])
        multihead_attn = nn.MultiheadAttention(128, 2)
        for param in multihead_attn.parameters():
            param.data = torch.from_numpy(np.random.random(param.shape).astype('float32'))
            param.requires_grad = True
        multihead_attn.eval()
        result = multihead_attn(query, key, value)
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
    )
