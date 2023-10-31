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

obj = APIBase("torch.nn.TransformerEncoder")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        transformer = nn.TransformerEncoder(layer, num_layers=6)
        tgt = torch.rand(20, 32, 512)
        result = transformer(tgt)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        transformer = nn.TransformerEncoder(layer, 6)
        tgt = torch.rand(20, 32, 512)
        result = transformer(tgt)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        transformer = nn.TransformerEncoder(encoder_layer=layer, num_layers=6)
        tgt = torch.rand(20, 32, 512)
        result = transformer(tgt)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        transformer = nn.TransformerEncoder(encoder_layer=layer, num_layers=6, norm=None)
        tgt = torch.rand(20, 32, 512)
        result = transformer(tgt)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        transformer = nn.TransformerEncoder(layer, 6, None)
        tgt = torch.rand(20, 32, 512)
        result = transformer(tgt)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        transformer = nn.TransformerEncoder(layer, 6, None, enable_nested_tensor=True)
        tgt = torch.rand(20, 32, 512)
        result = transformer(tgt)
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        check_value=False,
        unsupport=True,
        reason="paddle unsupport enable_nested_tensor args",
    )


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        transformer = nn.TransformerEncoder(layer, 6, None, mask_check=True)
        tgt = torch.rand(20, 32, 512)
        result = transformer(tgt)
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        check_value=False,
        unsupport=True,
        reason="paddle unsupport mask_check args",
    )
