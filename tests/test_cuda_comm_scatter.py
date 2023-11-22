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
#

import textwrap

from apibase import APIBase

obj = APIBase("torch.cuda.comm.scatter")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        nhwc = torch.randn((10, 3, 32, 32), device='cpu')
        results = torch.cuda.comm.scatter(tensor=nhwc)
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        unsupport=True,
        reason="paddle has no corresponding api tentatively",
    )


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        nhwc = torch.randn((10, 3, 32, 32), device='gpu')
        results = torch.cuda.comm.scatter(tensor=nhwc)
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        unsupport=True,
        reason="paddle has no corresponding api tentatively",
    )


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        nhwc = torch.randn((10, 3, 32, 32), device='gpu')
        devices = [torch.device('cuda:0'), torch.device('cuda:1')]
        result = torch.cuda.comm.scatter(nhwc, devices=devices)
    """
    )

    obj.run(
        pytorch_code,
        ["result"],
        unsupport=True,
        reason="paddle has no corresponding api tentatively",
    )


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        nhwc = torch.randn((10, 3, 32, 32), device='gpu')
        chunk_sizes = [5, 5]
        result = torch.cuda.comm.scatter(nhwc, chunk_sizes=chunk_sizes)
    """
    )

    obj.run(
        pytorch_code,
        ["result"],
        unsupport=True,
        reason="paddle has no corresponding api tentatively",
    )


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        nhwc = torch.randn((10, 3, 32, 32), device='gpu')
        result = torch.cuda.comm.scatter(nhwc, dim=1)
    """
    )

    obj.run(
        pytorch_code,
        ["result"],
        unsupport=True,
        reason="paddle has no corresponding api tentatively",
    )


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        nhwc = torch.randn((10, 3, 32, 32), device='gpu')
        t1 = torch.empty(5, 10, device='cuda')
        t2 = torch.empty(5, 10, device='cuda')
        result = torch.cuda.comm.scatter(nhwc, out=[t1, t2])
    """
    )

    obj.run(
        pytorch_code,
        ["result"],
        unsupport=True,
        reason="paddle has no corresponding api tentatively",
    )
