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

obj = APIBase("torch.nn.CrossEntropyLoss")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        loss = nn.CrossEntropyLoss()
        input = torch.ones(3, 5, requires_grad=True)
        target = torch.ones(3, dtype=torch.long)
        result = loss(input, target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        weight = torch.Tensor([1, 3, 4, 3, 2])
        input = torch.ones(3, 5)
        target = torch.ones(3, dtype=torch.long)
        loss = nn.CrossEntropyLoss(weight=weight)
        result = loss(input, target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        weight = torch.Tensor([1, 3, 4, 3, 2])
        input = torch.ones(3, 5)
        target = torch.ones(3, dtype=torch.long)
        loss = nn.CrossEntropyLoss(size_average=False, weight=weight)
        result = loss(input, target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input = torch.ones(3, 5)
        target = torch.ones(3, dtype=torch.long)
        loss = nn.CrossEntropyLoss(reduce=False)
        result = loss(input, target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        weight = torch.Tensor([1, 3, 4])
        input = torch.ones(3, 5)
        target = torch.ones(3, dtype=torch.long)
        loss = nn.CrossEntropyLoss(reduce='none')
        result = loss(input, target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        weight = torch.Tensor([1, 3, 4])
        input = torch.ones(3, 5)
        target = torch.ones(3, dtype=torch.long)
        loss = nn.CrossEntropyLoss(reduce='mean')
        result = loss(input, target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        weight = torch.Tensor([1, 3, 4])
        input = torch.ones(3, 5)
        target = torch.ones(3, dtype=torch.long)
        loss = nn.CrossEntropyLoss(reduce='sum')
        result = loss(input, target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        weight = torch.Tensor([1, 3, 4])
        input = torch.ones(3, 5)
        target = torch.ones(3, dtype=torch.long)
        loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        result = loss(input, target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        weight = torch.Tensor([1, 3, 4, 3, 2])
        input = torch.ones(3, 5)
        target = torch.ones(3, dtype=torch.long)
        loss = nn.CrossEntropyLoss(weight=weight, size_average=False, ignore_index=0, reduce=True, reduction='sum', label_smoothing=0.1)
        result = loss(input, target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_10():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        weight = torch.Tensor([1, 3, 4, 3, 2])
        input = torch.ones(3, 5)
        target = torch.ones(3, dtype=torch.long)
        loss = nn.CrossEntropyLoss(weight, False, 0, True, 'sum', 0.1)
        result = loss(input, target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_11():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        weight = torch.Tensor([1, 3, 4, 3, 2])
        input = torch.ones(3, 5)
        target = torch.ones(3, dtype=torch.long)
        loss = nn.CrossEntropyLoss(weight=weight, label_smoothing=0.1, reduce=True, size_average=False, ignore_index=0, reduction='sum')
        result = loss(input, target)
        """
    )
    obj.run(pytorch_code, ["result"])
