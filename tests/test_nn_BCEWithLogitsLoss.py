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

obj = APIBase("torch.nn.BCEWithLogitsLoss")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        input = torch.tensor([1.,0.7,0.2], requires_grad=True)
        target = torch.tensor([1.,0., 0.])
        result = loss(input, target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        loss = nn.BCEWithLogitsLoss(weight=torch.tensor([1.0,0.2, 0.2]), reduction='none')
        input = torch.tensor([1.,0.7,0.2], requires_grad=True)
        target = torch.tensor([1.,0., 0.])
        result = loss(input, target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        loss= nn.BCEWithLogitsLoss(pos_weight = torch.ones([3]))
        input = torch.tensor([1.,0.7,0.2], requires_grad=True)
        target = torch.tensor([1.,0., 0.])
        result = loss(input, target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        loss = nn.BCEWithLogitsLoss(size_average=True)
        input = torch.tensor([1.,0.7,0.2], requires_grad=True)
        target = torch.tensor([1.,0., 0.])
        result = loss(input, target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        loss = nn.BCEWithLogitsLoss()
        input = torch.tensor([1.,0.7,0.2], requires_grad=True)
        target = torch.tensor([1.,0., 0.])
        result = loss(input, target)
        """
    )
    obj.run(pytorch_code, ["result"])
