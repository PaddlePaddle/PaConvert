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

obj = APIBase("torch.nn.NLLLoss")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input = torch.arange(0, 15, dtype=torch.float32, requires_grad=True).reshape((3, 5))
        target = torch.tensor([1, 0, 4])
        m = nn.LogSoftmax(dim=1)
        loss = nn.NLLLoss(size_average=True, reduce=True, reduction="none")
        result = loss(m(input), target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input = torch.arange(0, 15, dtype=torch.float32, requires_grad=True).reshape((3, 5))
        target = torch.tensor([1, 0, 4])
        m = nn.LogSoftmax(dim=1)
        loss = nn.NLLLoss(size_average=True, reduce=False, reduction="none")
        result = loss(m(input), target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input = torch.arange(0, 15, dtype=torch.float32, requires_grad=True).reshape((3, 5))
        target = torch.tensor([1, 0, 4])
        m = nn.LogSoftmax(dim=1)
        loss = nn.NLLLoss(size_average=True, reduce=False, reduction="sum")
        result = loss(m(input), target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input = torch.arange(0, 15, dtype=torch.float32, requires_grad=True).reshape((3, 5))
        target = torch.tensor([1, 0, 4])
        m = nn.LogSoftmax(dim=1)
        loss = nn.NLLLoss(size_average=False, reduce=True, reduction="none")
        result = loss(m(input), target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input = torch.arange(0, 15, dtype=torch.float32, requires_grad=True).reshape((3, 5))
        target = torch.tensor([1, 0, 4])
        m = nn.LogSoftmax(dim=1)
        loss = nn.NLLLoss(size_average=True, reduce=True, reduction="mean")
        result = loss(m(input), target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input = torch.arange(0, 15, dtype=torch.float32, requires_grad=True).reshape((3, 5))
        target = torch.tensor([1, 0, 4])
        m = nn.LogSoftmax(dim=1)
        loss = nn.NLLLoss(size_average=True, reduce=True, reduction="sum")
        result = loss(m(input), target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input = torch.arange(0, 15, dtype=torch.float32, requires_grad=True).reshape((3, 5))
        target = torch.tensor([1, 0, 4])
        m = nn.LogSoftmax(dim=1)
        loss = nn.NLLLoss(size_average=False, reduce=True, reduction="mean")
        result = loss(m(input), target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input = torch.arange(0, 15, dtype=torch.float32, requires_grad=True).reshape((3, 5))
        target = torch.tensor([1, 0, 4])
        m = nn.LogSoftmax(dim=1)
        loss = nn.NLLLoss(size_average=False, reduce=True, reduction="sum")
        result = loss(m(input), target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input = torch.arange(0, 15, dtype=torch.float32, requires_grad=True).reshape((3, 5))
        target = torch.tensor([1, 0, 4])
        m = nn.LogSoftmax(dim=1)
        loss = nn.NLLLoss(size_average=False, reduce=False, reduction="none")
        result = loss(m(input), target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_10():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input = torch.arange(0, 15, dtype=torch.float32, requires_grad=True).reshape((3, 5))
        target = torch.tensor([1, 0, 4])
        m = nn.LogSoftmax(dim=1)
        loss = nn.NLLLoss(size_average=False, reduce=False, reduction="mean")
        result = loss(m(input), target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_11():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input = torch.arange(0, 15, dtype=torch.float32, requires_grad=True).reshape((3, 5))
        target = torch.tensor([1, 0, 4])
        m = nn.LogSoftmax(dim=1)
        loss = nn.NLLLoss(size_average=False, reduce=False, reduction="sum")
        result = loss(m(input), target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_12():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input = torch.arange(0, 15, dtype=torch.float32, requires_grad=True).reshape((3, 5))
        target = torch.tensor([1, 0, 4])
        m = nn.LogSoftmax(dim=1)
        loss = nn.NLLLoss(size_average=True, reduce=False, reduction="mean")
        result = loss(m(input), target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_13():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input = torch.arange(0, 15, dtype=torch.float32, requires_grad=True).reshape((3, 5))
        target = torch.tensor([1, 0, 4])
        m = nn.LogSoftmax(dim=1)
        loss = nn.NLLLoss(reduction="mean")
        result = loss(m(input), target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_14():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input = torch.arange(0, 15, dtype=torch.float32, requires_grad=True).reshape((3, 5))
        target = torch.tensor([1, 0, 4])
        m = nn.LogSoftmax(dim=1)
        loss = nn.NLLLoss(reduction="sum")
        result = loss(m(input), target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_15():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input = torch.arange(0, 15, dtype=torch.float32, requires_grad=True).reshape((3, 5))
        target = torch.tensor([1, 0, 4])
        m = nn.LogSoftmax(dim=1)
        loss = nn.NLLLoss(ignore_index=-1, reduction="sum")
        result = loss(m(input), target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_16():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input = torch.arange(0, 15, dtype=torch.float32, requires_grad=True).reshape((3, 5))
        target = torch.tensor([1, 0, 4])
        m = nn.LogSoftmax(dim=1)
        loss = nn.NLLLoss(None, True, -1, True, "mean")
        result = loss(m(input), target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_17():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input = torch.arange(0, 15, dtype=torch.float32, requires_grad=True).reshape((3, 5))
        target = torch.tensor([1, 0, 4])
        m = nn.LogSoftmax(dim=1)
        loss = nn.NLLLoss(weight=None, size_average=True, ignore_index=-1, reduce=True, reduction="mean")
        result = loss(m(input), target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_18():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input = torch.arange(0, 15, dtype=torch.float32, requires_grad=True).reshape((3, 5))
        target = torch.tensor([1, 0, 4])
        m = nn.LogSoftmax(dim=1)
        loss = nn.NLLLoss(reduce=True, reduction="mean", size_average=True)
        result = loss(m(input), target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_19():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input = torch.arange(0, 15, dtype=torch.float32, requires_grad=True).reshape((3, 5))
        target = torch.tensor([1, 0, 4])
        m = nn.LogSoftmax(dim=1)
        loss = nn.NLLLoss()
        result = loss(m(input), target)
        """
    )
    obj.run(pytorch_code, ["result"])
