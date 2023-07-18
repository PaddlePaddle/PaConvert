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

obj = APIBase("torch.nn.utils.clip_grad_value_")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        x = torch.tensor([[[[-0.4106,  0.1677], [-0.6648, -0.5669]]]])

        nn.utils.clip_grad_value_(x, clip_value=2.0)
        result = x
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        x = torch.tensor([[[[-0.4106,  0.1677], [-0.6648, -0.5669]]]])

        nn.utils.clip_grad_value_(x, clip_value=1.0)
        result = x
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        x = torch.tensor([[[[-0.4106,  0.1677], [-0.6648, -0.5669]]]])
        x.grad = torch.tensor([[[[-0.5, 12.343], [-10.4, -0.5669]]]])

        nn.utils.clip_grad_value_(x, clip_value=2)
        result = x.grad
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        x = torch.tensor([[[[-0.4106,  0.1677], [-0.6648, -0.5669]]]])
        x.grad = torch.tensor([[[[-0.5, 12.343], [-10.4, -0.5669]]]])

        nn.utils.clip_grad_value_(x, clip_value=0)
        result = x.grad
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        x = [torch.tensor([[[[-0.4106,  0.1677], [-0.6648, -0.5669]]]]), torch.tensor([[[[-0.4106,  0.1677], [-0.6648, -0.5669]]]])]
        x[0].grad = torch.tensor([[[[-0.5, 12.343], [-10.4, -0.5669]]]])
        x[1].grad = torch.tensor([[[[-0.67, 4.33], [-13, -0.69]]]])

        nn.utils.clip_grad_value_(x, clip_value=0.5)
        result1 = x[0].grad
        result2 = x[1].grad
        """
    )
    obj.run(pytorch_code, ["result1", "result2"])
