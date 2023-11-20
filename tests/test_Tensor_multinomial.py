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

obj = APIBase("torch.Tensor.multinomial")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        torch.manual_seed(100)
        weights = torch.tensor([0, 10, 3, 0], dtype=torch.float)
        result = weights.multinomial(2)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        torch.manual_seed(100)
        weights = torch.tensor([0, 10, 3, 0], dtype=torch.float)
        result = weights.multinomial(4, replacement=True)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        torch.manual_seed(100)
        result = torch.tensor([1., 10., 3., 2.]).multinomial(4, replacement=True)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        torch.manual_seed(100)
        weight = torch.tensor([[2., 4.], [4., 9.]])
        result = weight.multinomial(4, replacement=True)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        torch.manual_seed(100)
        weight = torch.tensor([[2., 4.], [4., 9.]])
        result = weight.multinomial(4, replacement=True, generator=None)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        torch.manual_seed(100)
        weight = torch.tensor([[2., 4.], [4., 9.]])
        result = weight.multinomial(num_samples=4, replacement=True, generator=None)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        torch.manual_seed(100)
        weight = torch.tensor([[2., 4.], [4., 9.]])
        result = weight.multinomial(4, True)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        torch.manual_seed(100)
        weight = torch.tensor([[2., 4.], [4., 9.]])
        result = weight.multinomial(replacement=True, generator=None, num_samples=4)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)
