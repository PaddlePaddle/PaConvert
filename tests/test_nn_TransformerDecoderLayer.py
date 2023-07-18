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

obj = APIBase("torch.nn.TransformerDecoderLayer")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        x = torch.ones(10, 32,512)
        tgt = torch.ones(10, 32, 512)
        model = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        result = model(tgt,x)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        x = torch.ones(10, 32,512)
        tgt = torch.ones(10, 32, 512)
        model = nn.TransformerDecoderLayer(d_model=512, nhead=8,norm_first=True)
        result = model(tgt,x)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        x = torch.ones(10, 32,512)
        tgt = torch.ones(10, 32, 512)
        model = nn.TransformerDecoderLayer(d_model=512, nhead=8,batch_first=True)
        result = model(tgt,x)
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        unsupport=True,
        reason="paddle unsupport batch_first args",
    )


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        x = torch.ones(10, 32,512)
        tgt = torch.ones(10, 32, 512)
        model = nn.TransformerDecoderLayer(d_model=512, nhead=8,dtype=torch.float32)
        result = model(tgt,x)
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        unsupport=True,
        reason="paddle not support astype args",
    )
