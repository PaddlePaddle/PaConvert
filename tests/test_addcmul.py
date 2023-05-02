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

import os
import sys

sys.path.append(os.path.dirname(__file__) + "/../")
import textwrap

from tests.apibase import APIBase

obj = APIBase("torch.addcmul")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        tensor1 = torch.tensor([1., 2., 3.])
        tensor2 = torch.tensor([4., 5., 6.])
        input = torch.tensor([7., 8., 9.])
        out = torch.tensor([1., 1., 1.])
        result = torch.addcmul(input, tensor1, tensor2, out=out)
        """
    )
    obj.run(pytorch_code, ["out"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        tensor1 = torch.tensor([1., 2., 3.])
        tensor2 = torch.tensor([4., 5., 6.])
        input = torch.tensor([7., 8., 9.])
        result = torch.addcmul(input, tensor1, tensor2)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        tensor1 = torch.tensor([1., 2., 3.])
        tensor2 = torch.tensor([4., 5., 6.])
        input = torch.tensor([7., 8., 9.])
        result = torch.addcmul(input, tensor1, tensor2, value=2)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        tensor1 = torch.tensor([1., 2., 3.])
        tensor2 = torch.tensor([4., 5., 6.])
        input = torch.tensor([7., 8., 9.])
        value = 5.0
        result = torch.addcmul(input, tensor1, tensor2=tensor2, value=value)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        tensor1 = torch.tensor([1., 2., 3.])
        tensor2 = torch.tensor([4., 5., 6.])
        input = torch.tensor([7., 8., 9.])
        value = 5
        result = torch.addcmul(input, tensor1, tensor2, value=value)
        """
    )
    obj.run(pytorch_code, ["result"])
