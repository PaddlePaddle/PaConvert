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

obj = APIBase("torch.distributions.transforms.ReshapeTransform")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        t = torch.distributions.transforms.ReshapeTransform(torch.Size([4]), torch.Size([2, 2]))
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = t(x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        t = torch.distributions.transforms.ReshapeTransform(in_shape=torch.Size([2, 2]), out_shape=torch.Size([4]), cache_size=0)
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = t(x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        t = torch.distributions.transforms.ReshapeTransform(torch.Size([4]), torch.Size([2, 2]), 0)
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = t(x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        t = torch.distributions.transforms.ReshapeTransform(in_shape=torch.Size([2, 2]), out_shape=torch.Size([4]))
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = t(x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    """Keyword arguments out of order test"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        t = torch.distributions.transforms.ReshapeTransform(out_shape=torch.Size([4]), in_shape=torch.Size([2, 2]), cache_size=0)
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = t(x)
        """
    )
    obj.run(pytorch_code, ["result"])
