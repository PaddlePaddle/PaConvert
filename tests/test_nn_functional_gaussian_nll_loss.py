# Copyright (c) 2023 torchtorch Authors. All Rights Reserved.
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

obj = APIBase("torch.nn.functional.gaussian_nll_loss")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.ones([5, 2], dtype=torch.float32)
        label = torch.ones([5, 2], dtype=torch.float32)
        variance = torch.ones([5, 2], dtype=torch.float32)
        result = torch.nn.functional.gaussian_nll_loss(input, label, variance)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.ones([5, 2], dtype=torch.float32)
        label = torch.ones([5, 2], dtype=torch.float32)
        variance = torch.ones([5, 2], dtype=torch.float32)
        result = torch.nn.functional.gaussian_nll_loss(input=input, target=label, var=variance)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.ones([5, 2], dtype=torch.float32)
        label = torch.ones([5, 2], dtype=torch.float32)
        variance = torch.ones([5, 2], dtype=torch.float32)
        result = torch.nn.functional.gaussian_nll_loss(target=label, var=variance, input=input)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.ones([5, 2], dtype=torch.float32)
        label = torch.ones([5, 2], dtype=torch.float32)
        variance = torch.ones([5, 2], dtype=torch.float32)
        result = torch.nn.functional.gaussian_nll_loss(input=input, target=label, var=variance, full=False, eps=1e-06, reduction='mean')
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.ones([5, 2], dtype=torch.float32)
        label = torch.ones([5, 2], dtype=torch.float32)
        variance = torch.ones([5, 2], dtype=torch.float32)
        result = torch.nn.functional.gaussian_nll_loss(input, label, variance, False, 1e-06, 'mean')
        """
    )
    obj.run(pytorch_code, ["result"])
