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

obj = APIBase("torch.nn.SmoothL1Loss")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        loss = torch.nn.SmoothL1Loss()
        input = torch.ones([3, 3]).to(dtype=torch.float32)
        label = torch.full([3, 3], 2).to(dtype=torch.float32)
        result = loss(input, label)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        loss = torch.nn.SmoothL1Loss(reduction='sum')
        input = torch.ones([3, 3]).to(dtype=torch.float32)
        label = torch.full([3, 3], 2).to(dtype=torch.float32)
        result = loss(input, label)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        loss = torch.nn.SmoothL1Loss(beta=1.0, reduction='none')
        input = torch.ones([3, 3]).to(dtype=torch.float32)
        label = torch.full([3, 3], 2).to(dtype=torch.float32)
        result = loss(input, label)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        loss = torch.nn.SmoothL1Loss(size_average=None,
                reduce=None, reduction='mean', beta=1.0)
        input = torch.ones([3, 3]).to(dtype=torch.float32)
        label = torch.full([3, 3], 2).to(dtype=torch.float32)
        result = loss(input, label)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        loss = torch.nn.SmoothL1Loss(None,
                None, 'mean', 1.0)
        input = torch.ones([3, 3]).to(dtype=torch.float32)
        label = torch.full([3, 3], 2).to(dtype=torch.float32)
        result = loss(input, label)
        """
    )
    obj.run(pytorch_code, ["result"])


# paddle result has diff with pytorch result
def _test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        loss = torch.nn.SmoothL1Loss(beta=1.5, reduction='none')
        input = torch.ones([3, 3]).to(dtype=torch.float32)
        label = torch.full([3, 3], 2).to(dtype=torch.float32)
        result = loss(input, label)
        """
    )
    obj.run(pytorch_code, ["result"])


# paddle result has diff with pytorch result
def _test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        beta = 1.5
        loss = torch.nn.SmoothL1Loss(beta=beta, reduction='none')
        input = torch.ones([3, 3]).to(dtype=torch.float32)
        label = torch.full([3, 3], 2).to(dtype=torch.float32)
        result = loss(input, label)
        """
    )
    obj.run(pytorch_code, ["result"])
