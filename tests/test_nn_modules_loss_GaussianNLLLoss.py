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

obj = APIBase("torch.nn.modules.loss.GaussianNLLLoss")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        loss = torch.nn.modules.loss.GaussianNLLLoss()
        input = torch.ones([3, 5]).to(dtype=torch.float32)
        label = torch.ones([3, 5]).to(dtype=torch.float32)
        variance = torch.ones([3, 5]).to(dtype=torch.float32)
        result = loss(input, label, variance)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        loss = torch.nn.modules.loss.GaussianNLLLoss(full=True)
        input = torch.ones([3, 5]).to(dtype=torch.float32)
        label = torch.ones([3, 5]).to(dtype=torch.float32)
        variance = torch.ones([3, 5]).to(dtype=torch.float32)
        result = loss(input, label, variance)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        loss = torch.nn.modules.loss.GaussianNLLLoss(eps=1e-06)
        input = torch.ones([3, 5]).to(dtype=torch.float32)
        label = torch.ones([3, 5]).to(dtype=torch.float32)
        variance = torch.ones([3, 5]).to(dtype=torch.float32)
        result = loss(input, label, variance)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        loss = torch.nn.modules.loss.GaussianNLLLoss(reduction='none')
        input = torch.ones([3, 5]).to(dtype=torch.float32)
        label = torch.ones([3, 5]).to(dtype=torch.float32)
        variance = torch.ones([3, 5]).to(dtype=torch.float32)
        result = loss(input, label, variance)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        loss = torch.nn.modules.loss.GaussianNLLLoss(reduction='mean')
        input = torch.ones([3, 5]).to(dtype=torch.float32)
        label = torch.ones([3, 5]).to(dtype=torch.float32)
        variance = torch.ones([3, 5]).to(dtype=torch.float32)
        result = loss(input, label, variance)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        loss = torch.nn.modules.loss.GaussianNLLLoss(reduction='sum')
        input = torch.ones([3, 5]).to(dtype=torch.float32)
        label = torch.ones([3, 5]).to(dtype=torch.float32)
        variance = torch.ones([3, 5]).to(dtype=torch.float32)
        result = loss(input, label, variance)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        loss = torch.nn.modules.loss.GaussianNLLLoss(full=True, eps=1e-06, reduction='mean')
        input = torch.ones([3, 5]).to(dtype=torch.float32)
        label = torch.ones([3, 5]).to(dtype=torch.float32)
        variance = torch.ones([3, 5]).to(dtype=torch.float32)
        result = loss(input, label, variance)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        loss = torch.nn.modules.loss.GaussianNLLLoss(full=False, eps=1e-08, reduction='sum')
        input = torch.full([3, 5], 1).to(dtype=torch.float32)
        label = torch.full([3, 5], 2).to(dtype=torch.float32)
        variance = torch.ones([3, 5]).to(dtype=torch.float32)
        result = loss(input, label, variance)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    pytorch_code = textwrap.dedent(
        """
        import torch
        loss = torch.nn.modules.loss.GaussianNLLLoss(reduction='none', full=True, eps=1e-06)
        input = torch.full([3, 5], 1).to(dtype=torch.float32)
        label = torch.full([3, 5], 2).to(dtype=torch.float32)
        variance = torch.ones([3, 5]).to(dtype=torch.float32)
        result = loss(input, label, variance)
        """
    )
    obj.run(pytorch_code, ["result"])
