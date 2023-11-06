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

obj = APIBase("torch.nn.PoissonNLLLoss")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        loss = torch.nn.PoissonNLLLoss()
        input = torch.ones([5, 2]).to(dtype=torch.float32)
        label = torch.ones([5, 2]).to(dtype=torch.float32)
        result = loss(input, label)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        loss = torch.nn.PoissonNLLLoss(log_input=True, full=False)
        input = torch.ones([5, 2]).to(dtype=torch.float32)
        label = torch.ones([5, 2]).to(dtype=torch.float32)
        result = loss(input, label)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        loss = torch.nn.PoissonNLLLoss(log_input=True, eps=1e-08,
                size_average=None, full=False)
        input = torch.full([5, 2], 1).to(dtype=torch.float32)
        label = torch.full([5, 2], 2).to(dtype=torch.float32)
        result = loss(input, label)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        loss = torch.nn.PoissonNLLLoss(log_input=True, full=False,
                size_average=None, eps=1e-08,
                reduce=None, reduction='mean')
        input = torch.full([5, 2], 1).to(dtype=torch.float32)
        label = torch.full([5, 2], 2).to(dtype=torch.float32)
        result = loss(input, label)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        loss = torch.nn.PoissonNLLLoss(True, False,
                None, 1e-08,
                None, 'sum')
        input = torch.full([5, 2], 1).to(dtype=torch.float32)
        label = torch.full([5, 2], 2).to(dtype=torch.float32)
        result = loss(input, label)
        """
    )
    obj.run(pytorch_code, ["result"])
