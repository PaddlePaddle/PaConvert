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

obj = APIBase("torch.nn.functional.interpolate")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.tensor([[[[1., 2., 3.], [2., 3., 4.]]]])
        result = F.interpolate(x, scale_factor=2)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.tensor([[[[1., 2., 3.], [2., 3., 4.]]]])
        result = F.interpolate(x, size=(2,3))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.tensor([[[[1., 2., 3.], [2., 3., 4.]]]])
        result = F.interpolate(x, scale_factor=2)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.tensor([[[[1., 2., 3.], [2., 3., 4.]]]])
        result = F.interpolate(x, scale_factor=2, mode='nearest')
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.tensor([[[[1., 2., 3.], [2., 3., 4.]]]])
        result = F.interpolate(x, scale_factor=4, mode='nearest')
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F

        x = torch.tensor([[[1., 2., 3.], [2., 3., 4.]]])
        result = F.interpolate(input=x, size=None, scale_factor=3, mode='linear', align_corners=False,
                                recompute_scale_factor=True, antialias=False)
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        unsupport=True,
        reason="paddle unsupport parameter antialias",
    )


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F

        x = torch.tensor([[[1., 2., 3.], [2., 3., 4.]]])
        result = F.interpolate(input=x, scale_factor=3, size=None, recompute_scale_factor=True, mode='linear', align_corners=False,
                                antialias=False)
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        unsupport=True,
        reason="paddle unsupport parameter antialias",
    )


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F

        x = torch.tensor([[[1., 2., 3.], [2., 3., 4.]]])
        result = F.interpolate(x, None, 3, 'linear', False, True, False)
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        unsupport=True,
        reason="paddle unsupport parameter antialias",
    )


def test_case_9():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F

        x = torch.tensor([[[1., 2., 3.], [2., 3., 4.]]])
        result = F.interpolate(input=x, scale_factor=3, size=None, recompute_scale_factor=True, mode='linear', align_corners=False)
        """
    )
    obj.run(pytorch_code, ["result"])
