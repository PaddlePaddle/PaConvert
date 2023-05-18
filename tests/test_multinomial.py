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


class MultinomialAPI(APIBase):
    def __init__(self, pytorch_api) -> None:
        super().__init__(pytorch_api)

    def check(self, pytorch_result, paddle_result):
        if pytorch_result.requires_grad == paddle_result.stop_gradient:
            return False
        if str(pytorch_result.dtype)[6:] != str(paddle_result.dtype)[7:]:
            return False
        if pytorch_result.numpy().shape != paddle_result.numpy().shape:
            return False
        return True


obj = MultinomialAPI("torch.multinomial")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        torch.manual_seed(100)
        weights = torch.tensor([0, 10, 3, 0], dtype=torch.float)
        result = torch.multinomial(weights, 2)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        torch.manual_seed(100)
        weights = torch.tensor([0, 10, 3, 0], dtype=torch.float)
        result = torch.multinomial(weights, 4, replacement=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        torch.manual_seed(100)
        result = torch.multinomial(torch.tensor([1., 10., 3., 2.]), 4, replacement=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        torch.manual_seed(100)
        weight = torch.tensor([[2., 4.], [4., 9.]])
        result = torch.multinomial(weight, 4, replacement=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        torch.manual_seed(100)
        weight = torch.tensor([[2., 4.], [4., 9.]])
        out = torch.zeros(2, 4, dtype=torch.int64)
        result = torch.multinomial(weight, 4, replacement=True, out=out)
        """
    )
    obj.run(pytorch_code, ["result", "out"])
