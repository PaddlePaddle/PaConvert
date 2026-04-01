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

obj = APIBase("torch.nn.ConvTranspose3d")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        x = torch.rand(2, 16, 50, 20, 20)
        model = nn.ConvTranspose3d(16, 33, 3, stride=2, bias=False)
        result = model(x)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        x = torch.randn(2, 16, 50, 20, 20)
        model = nn.ConvTranspose3d(16, 33, (3, 3, 5), stride=(2, 2, 1), padding=(4, 2, 2), bias=False)
        result = model(x)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        x = torch.randn(2, 16, 50, 20, 20)
        model = nn.ConvTranspose3d(16, 33, (3, 3, 5), stride=(2, 2, 1), padding=(4, 3, 2), dilation=(3, 1, 1), bias=False)
        result = model(x)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        x = torch.randn(5, 16, 50, 20, 20)
        model = nn.ConvTranspose3d(16, 33, (3, 3, 5), stride=(2, 2, 1), padding=(4, 2, 2), dilation=(3, 1, 1), bias=True)
        result = model(x)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        x = torch.randn(5, 16, 50, 20, 20)
        model = nn.ConvTranspose3d(16, 33, (3, 3, 5), stride=(2, 2, 1), padding=(4, 2, 2), dilation=(3, 1, 1), bias=True, padding_mode='zeros')
        result = model(x)
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        check_value=False,
    )


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        x = torch.randn(5, 16, 50, 20, 20)
        model = nn.ConvTranspose3d(in_channels=16, out_channels=33, kernel_size=(3, 3, 5), stride=(2, 2, 1), padding=(4, 2, 2), output_padding=0, groups=1, bias=True, dilation=(3, 1, 1), padding_mode='zeros', device=None, dtype=None)
        result = model(x)
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        check_value=False,
    )


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        x = torch.randn(5, 16, 50, 20, 20)
        model = nn.ConvTranspose3d(in_channels=16, kernel_size=(3, 3, 5), out_channels=33, stride=(2, 2, 1), device=None, padding=(4, 2, 2), bias=True, output_padding=0, groups=1, dilation=(3, 1, 1), padding_mode='zeros', dtype=None)
        result = model(x)
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        check_value=False,
    )


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        x = torch.randn(5, 16, 50, 20, 20)
        model = nn.ConvTranspose3d(16, 33, (3, 3, 5), (2, 2, 1), (4, 2, 2), 0, 1, True, (3, 1, 1), 'zeros', None, None)
        result = model(x)
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        check_value=False,
    )


def test_case_9():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        x = torch.randn(5, 16, 50, 20, 20)
        model = nn.ConvTranspose3d(16, 33, (3, 3, 5))
        result = model(x)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_10():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn

        conv_args = (8, 6, (3, 3, 2))
        conv_kwargs = {
            "stride": (2, 1, 1),
            "padding": (1, 1, 0),
            "output_padding": (1, 0, 0),
            "groups": 2,
            "bias": False,
            "dilation": (1, 1, 1),
        }
        model = nn.ConvTranspose3d(*conv_args, **conv_kwargs)
        x = torch.randn(2, 8, 5, 6, 4)
        result = model(x)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_11():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn

        model = nn.ConvTranspose3d(
            dilation=(1, 1, 1),
            groups=1,
            output_padding=(1, 0, 0),
            padding=(1, 1, 0),
            stride=(2, 1, 1),
            kernel_size=(3, 3, 2),
            out_channels=10,
            in_channels=8,
            bias=True,
        )
        x = torch.randn(2, 8, 5, 6, 4)
        result = model(x)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_12():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn

        model = nn.ConvTranspose3d(8, 10, (3, 3, 2), (2, 1, 1), padding=(1, 1, 0), output_padding=(1, 0, 0), bias=False)
        x = torch.randn(2, 8, 5, 6, 4)
        result = model(x)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_13():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn

        model = nn.ConvTranspose3d(8, 10, (3, 3, 2), (2, 1, 1), (1, 1, 0), (1, 0, 0), 1, True, (1, 1, 1))
        x = torch.randn(2, 8, 5, 6, 4)
        result = model(x)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_14():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn

        model = nn.ConvTranspose3d(4, 5, 3, stride=2, padding=1).double()
        x = torch.randn(2, 4, 4, 4, 4, dtype=torch.float64)
        result = model(x)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)

