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

obj = APIBase("torch.special.log_softmax")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([1.4907, 1.0593, 1.5696])
        result = torch.special.log_softmax(input, 0)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[1.4907, 1.0593, 1.5696], [1.4907, 1.0593, 1.5696]])
        result = torch.special.log_softmax(input, dim=1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[1.4907, 1.0593, 1.5696], [1.4907, 1.0593, 1.5696]])
        result = torch.special.log_softmax(input, 1, dtype=torch.float32)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[1.4907, 1.0593, 1.5696], [1.4907, 1.0593, 1.5696]])
        result = torch.special.log_softmax(input=input, dim=1, dtype=torch.float32)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[1.4907, 1.0593, 1.5696], [1.4907, 1.0593, 1.5696]])
        result = torch.special.log_softmax(dim=1, input=input, dtype=torch.float32)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    # all defaults omitted (no dtype)
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[1.4907, 1.0593, 1.5696], [1.4907, 1.0593, 1.5696]])
        result = torch.special.log_softmax(input, 0)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    # negative dim
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[1.4907, 1.0593, 1.5696], [1.4907, 1.0593, 1.5696]])
        result = torch.special.log_softmax(input, dim=-1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    # 1D input
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([1.4907, 1.0593, 1.5696])
        result = torch.special.log_softmax(input, 0)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    # 3D input
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        result = torch.special.log_softmax(input, dim=2)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_10():
    # gradient computation
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
        result = torch.special.log_softmax(input, dim=1)
        result.sum().backward()
        input_grad = input.grad
        """
    )
    obj.run(pytorch_code, ["result", "input_grad"], check_stop_gradient=False)


def test_case_11():
    # variable args
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[1.4907, 1.0593, 1.5696], [1.4907, 1.0593, 1.5696]])
        args = (input, 1)
        result = torch.special.log_softmax(*args)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_12():
    # kwargs dict unpacking
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[1.4907, 1.0593, 1.5696], [1.4907, 1.0593, 1.5696]])
        kwargs = {'input': input, 'dim': 1, 'dtype': torch.float64}
        result = torch.special.log_softmax(**kwargs)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_13():
    # float64 input
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64)
        result = torch.special.log_softmax(input, dim=1)
        """
    )
    obj.run(pytorch_code, ["result"])
