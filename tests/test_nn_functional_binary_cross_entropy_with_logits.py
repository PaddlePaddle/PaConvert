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

obj = APIBase("torch.nn.functional.binary_cross_entropy_with_logits")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        input = torch.tensor([ 2.5036,  1.2420, -0.5798])
        target = torch.tensor([0., 1., 0.])
        result = F.binary_cross_entropy_with_logits(input, target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        input = torch.tensor([ 2.5036,  1.2420, -0.5798])
        weight = torch.tensor([1., 3., 3.])
        target = torch.tensor([0., 1., 0.])
        result = F.binary_cross_entropy_with_logits(input, target, weight=weight)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        input = torch.tensor([[ 2.5036,  1.2420, -0.5798],[ 2.5036,  1.2420, -1.5798]])
        target = torch.tensor([[0., 1., 0.],[0., 1., 0.]])
        result = F.binary_cross_entropy_with_logits(input, target, reduce=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        input = torch.tensor([[ 2.5036,  1.2420, -0.5798],[ 2.5036,  1.2420, -1.5798]])
        target = torch.tensor([[0., 1., 0.],[0., 1., 0.]])
        result = F.binary_cross_entropy_with_logits(input, target, reduce="none")
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        input = torch.tensor([[ 2.5036,  1.2420, -0.5798],[ 2.5036,  1.2420, -1.5798]])
        target = torch.tensor([[0., 1., 0.],[0., 1., 0.]])
        result = F.binary_cross_entropy_with_logits(input, target, reduce="mean")
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        input = torch.tensor([[ 2.5036,  1.2420, -0.5798],[ 2.5036,  1.2420, -1.5798]])
        target = torch.tensor([[0., 1., 0.],[0., 1., 0.]])
        result = F.binary_cross_entropy_with_logits(input, target, reduce="sum")
        print(result)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        input = torch.tensor([[ 2.5036,  1.2420, -0.5798],[ 2.5036,  1.2420, -1.5798]])
        target = torch.tensor([[0., 1., 0.],[0., 1., 0.]])
        result = F.binary_cross_entropy_with_logits(input, target, reduce=False)
        """
    )
    obj.run(pytorch_code, ["result"])
