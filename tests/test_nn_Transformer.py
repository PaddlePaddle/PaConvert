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

obj = APIBase("torch.nn.Transformer")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        transformer_model = torch.nn.Transformer()
        src = torch.rand((10, 32, 512))
        tgt = torch.rand((10, 32, 512))
        result = transformer_model(src, tgt)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        transformer_model = torch.nn.Transformer(d_model=512,
                nhead=8, num_encoder_layers=6,
                num_decoder_layers=6, dim_feedforward=2048,
                dropout=0.1, activation='relu'
                )
        src = torch.rand((10, 32, 512))
        tgt = torch.rand((10, 32, 512))
        result = transformer_model(src, tgt)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        transformer_model = torch.nn.Transformer(d_model=512,
                num_decoder_layers=6, dim_feedforward=2048,
                dropout=0.1, activation='relu',
                custom_encoder=None, custom_decoder=None,
                nhead=8, num_encoder_layers=6)
        src = torch.rand((10, 32, 512))
        tgt = torch.rand((10, 32, 512))
        result = transformer_model(src, tgt)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        transformer_model = torch.nn.Transformer(d_model=512,
                nhead=8, num_encoder_layers=6,
                num_decoder_layers=6, dim_feedforward=2048,
                dropout=0.1, activation='relu',
                custom_encoder=None, custom_decoder=None,
                norm_first=False, device=None, dtype=None)
        src = torch.rand((10, 32, 512))
        tgt = torch.rand((10, 32, 512))
        result = transformer_model(src, tgt)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        transformer_model = torch.nn.Transformer(512,
                8, 6,
                6, 2048,
                0.1, 'relu',
                None, None)
        src = torch.rand((10, 32, 512))
        tgt = torch.rand((10, 32, 512))
        result = transformer_model(src, tgt)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        transformer_model = torch.nn.Transformer(512,
                8, 6,
                6, 2048,
                0.1, 'relu',
                None, None,
                layer_norm_eps=1e-05)
        src = torch.rand((10, 32, 512))
        tgt = torch.rand((10, 32, 512))
        result = transformer_model(src, tgt)
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        check_value=False,
        unsupport=True,
        reason="paddle unsupport layer_norm_eps args",
    )


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        transformer_model = torch.nn.Transformer(512,
                8, 6,
                6, 2048,
                0.1, 'relu',
                None, None,
                batch_first=False
                )
        src = torch.rand((10, 32, 512))
        tgt = torch.rand((10, 32, 512))
        result = transformer_model(src, tgt)
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        check_value=False,
        unsupport=True,
        reason="paddle unsupport batch_first args",
    )


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        transformer_model = torch.nn.Transformer(d_model=512,
                nhead=8, num_encoder_layers=6,
                num_decoder_layers=6, dim_feedforward=2048,
                dropout=0.1, activation='relu',
                custom_encoder=None, custom_decoder=None,
                layer_norm_eps=1e-05, batch_first=False,
                norm_first=False, bias=False,
                device=None, dtype=None)
        src = torch.rand((10, 32, 512))
        tgt = torch.rand((10, 32, 512))
        result = transformer_model(src, tgt)
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        check_value=False,
        unsupport=True,
        reason="paddle unsupport layer_norm_eps args",
    )


def test_case_9():
    pytorch_code = textwrap.dedent(
        """
        import torch
        transformer_model = torch.nn.Transformer(512,
                8, 6, 6, 2048,
                0.1, 'relu',
                None, None,
                1e-05, False,
                False, False,
                None, None)
        src = torch.rand((10, 32, 512))
        tgt = torch.rand((10, 32, 512))
        result = transformer_model(src, tgt)
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        check_value=False,
        unsupport=True,
        reason="paddle unsupport layer_norm_eps args",
    )
