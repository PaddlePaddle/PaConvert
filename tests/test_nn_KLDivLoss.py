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

obj = APIBase("torch.nn.KLDivLoss")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input = torch.tensor([[-1.2837, -0.0297,  0.0355],
            [ 0.9112, -1.7526, -0.4061]])
        target = torch.tensor([[1.,2.,3.],[4.,5.,6.]])
        loss = torch.nn.KLDivLoss(size_average=True)
        result = loss(input,target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input = torch.tensor([[-1.2837, -0.0297,  0.0355],
            [ 0.9112, -1.7526, -0.4061]])
        target = torch.tensor([[1.,2.,3.],[4.,5.,6.]])
        loss = torch.nn.KLDivLoss(size_average=False)
        result = loss(input,target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input = torch.tensor([[-1.2837, -0.0297,  0.0355],
            [ 0.9112, -1.7526, -0.4061]])
        target = torch.tensor([[1.,2.,3.],[4.,5.,6.]])
        loss = torch.nn.KLDivLoss(reduction='none')
        result = loss(input,target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input = torch.tensor([[-1.2837, -0.0297,  0.0355],
            [ 0.9112, -1.7526, -0.4061]])
        target = torch.tensor([[1.,2.,3.],[4.,5.,6.]])
        loss = torch.nn.KLDivLoss(reduction='mean')
        result = loss(input,target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input = torch.tensor([[-1.2837, -0.0297,  0.0355],
            [ 0.9112, -1.7526, -0.4061]])
        target = torch.tensor([[1.,2.,3.],[4.,5.,6.]])
        loss = torch.nn.KLDivLoss(reduction='sum')
        result = loss(input,target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input = torch.tensor([[-1.2837, -0.0297,  0.0355],
            [ 0.9112, -1.7526, -0.4061]])
        target = torch.tensor([[1.,2.,3.],[4.,5.,6.]])
        loss = torch.nn.KLDivLoss(reduce=True)
        result = loss(input,target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input = torch.tensor([[-1.2837, -0.0297,  0.0355],
            [ 0.9112, -1.7526, -0.4061]])
        target = torch.tensor([[1.,2.,3.],[4.,5.,6.]])
        loss = torch.nn.KLDivLoss(reduce=False)
        result = loss(input,target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input = torch.tensor([[-1.2837, -0.0297,  0.0355],
            [ 0.9112, -1.7526, -0.4061]])
        target = torch.tensor([[1.,2.,3.],[4.,5.,6.]])
        loss = torch.nn.KLDivLoss()
        result = loss(input,target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input = torch.tensor([[-1.2837, -0.0297,  0.0355],
            [ 0.9112, -1.7526, -0.4061]])
        target = torch.tensor([[1.,2.,3.],[4.,5.,6.]])
        loss = torch.nn.KLDivLoss(log_target=False)
        result = loss(input,target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_10():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input = torch.tensor([[-1.2837, -0.0297,  0.0355],
            [ 0.9112, -1.7526, -0.4061]])
        target = torch.tensor([[1.,2.,3.],[4.,5.,6.]])
        loss = torch.nn.KLDivLoss(size_average=False, reduce=True, reduction='batchmean', log_target=False)
        result = loss(input,target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_11():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input = torch.tensor([[-1.2837, -0.0297,  0.0355],
            [ 0.9112, -1.7526, -0.4061]])
        target = torch.tensor([[1.,2.,3.],[4.,5.,6.]])
        loss = torch.nn.KLDivLoss(False, True, 'batchmean', False)
        result = loss(input,target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_12():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input = torch.tensor([[-1.2837, -0.0297,  0.0355],
            [ 0.9112, -1.7526, -0.4061]])
        target = torch.tensor([[1.,2.,3.],[4.,5.,6.]])
        loss = torch.nn.KLDivLoss(reduction='batchmean', log_target=False, size_average=False, reduce=True, )
        result = loss(input,target)
        """
    )
    obj.run(pytorch_code, ["result"])
