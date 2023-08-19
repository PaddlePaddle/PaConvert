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

obj = APIBase("torch.nn.functional.binary_cross_entropy_with_logits")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.zeros(3, requires_grad=True)
        target = torch.tensor([0.,1.,0.])
        result = torch.nn.functional.binary_cross_entropy_with_logits(a, target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.zeros(3, requires_grad=True)
        target = torch.tensor([0.,1.,0.])
        result = torch.nn.functional.binary_cross_entropy_with_logits(a, target, weight=None)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.zeros(3, requires_grad=True)
        target = torch.tensor([0.,1.,0.])
        result = torch.nn.functional.binary_cross_entropy_with_logits(a, target, reduce=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.zeros(3, requires_grad=True)
        target = torch.tensor([0.,1.,0.])
        result = torch.nn.functional.binary_cross_entropy_with_logits(a, target, reduce=False)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.zeros(3, requires_grad=True)
        target = torch.tensor([0.,1.,0.])
        result = torch.nn.functional.binary_cross_entropy_with_logits(a, target, reduction='none')
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.zeros(3, requires_grad=True)
        target = torch.tensor([0.,1.,0.])
        result = torch.nn.functional.binary_cross_entropy_with_logits(a, target, reduction='mean')
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.zeros(3, requires_grad=True)
        target = torch.tensor([0.,1.,0.])
        result = torch.nn.functional.binary_cross_entropy_with_logits(a, target, reduction='sum')
        """
    )
    obj.run(pytorch_code, ["result"])
