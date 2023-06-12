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

obj = APIBase("torch.nn.functional.max_unpool3d")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.tensor([[[[[0.9064, 1.9579], [1.6145, 1.0924]], [[1.1187, 1.9942], [1.1927, 0.8372]]]]])
        indices = torch.tensor([[[[[ 5,  2], [ 8, 30]], [[37, 54], [61, 46]]]]])
        result = F.max_unpool3d(x, indices, kernel_size=2, padding=0)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.tensor([[[[[0.9064, 1.9579], [1.6145, 1.0924]], [[1.1187, 1.9942], [1.1927, 0.8372]]]]])
        indices = torch.tensor([[[[[ 5,  2], [ 8, 30]], [[37, 54], [61, 46]]]]])
        result = F.max_unpool3d(x, indices, kernel_size=2, stride=None)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.tensor([[[[[0.9064, 1.9579], [1.6145, 1.0924]], [[1.1187, 1.9942], [1.1927, 0.8372]]]]])
        indices = torch.tensor([[[[[ 5,  2], [ 8, 30]], [[37, 54], [61, 46]]]]])
        result = F.max_unpool3d(x, indices, kernel_size=2, stride=2, padding=0)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.tensor([[[[[0.9064, 1.9579], [1.6145, 1.0924]], [[1.1187, 1.9942], [1.1927, 0.8372]]]]])
        indices = torch.tensor([[[[[ 5,  2], [ 8, 30]], [[37, 54], [61, 46]]]]])
        result = F.max_unpool3d(x, indices, kernel_size=2, output_size=(1, 1, 4, 4, 4))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.tensor([[[[[0.9064, 1.9579], [1.6145, 1.0924]], [[1.1187, 1.9942], [1.1927, 0.8372]]]]])
        indices = torch.tensor([[[[[ 5,  2], [ 8, 30]], [[37, 54], [61, 46]]]]])
        result = F.max_unpool3d(x, indices, kernel_size=2, padding=0)
        """
    )
    obj.run(pytorch_code, ["result"])
