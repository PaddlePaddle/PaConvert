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

obj = APIBase("torch.log_softmax")


def test_case_1():
    # positional input and dim
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = torch.log_softmax(x, 1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    # keyword dim
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = torch.log_softmax(x, dim=1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    # all keyword args
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = torch.log_softmax(input=x, dim=1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    # dim=0
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = torch.log_softmax(x, dim=0)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    # negative dim
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = torch.log_softmax(x, dim=-1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    # with dtype
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = torch.log_softmax(x, 1, dtype=torch.float64)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    # all kwargs with dtype, out-of-order
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = torch.log_softmax(dtype=torch.float64, input=x, dim=0)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    # 3D input
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        result = torch.log_softmax(x, dim=2)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    # 3D input, negative dim
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        result = torch.log_softmax(x, dim=-2)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_10():
    # 1D input
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = torch.log_softmax(x, 0)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_11():
    # float64 input
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64)
        result = torch.log_softmax(x, dim=1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_12():
    # gradient computation
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
        result = torch.log_softmax(x, dim=1)
        result.sum().backward()
        x_grad = x.grad
        """
    )
    obj.run(pytorch_code, ["result", "x_grad"], check_stop_gradient=False)


def test_case_13():
    # variable args
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        args = (x, 1)
        result = torch.log_softmax(*args)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_14():
    # all positional parameters including dtype
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = torch.log_softmax(x, 1, torch.float64)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_15():
    # all keyword parameters including dtype
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = torch.log_softmax(input=x, dim=1, dtype=torch.float64)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_16():
    # kwargs dict unpacking
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        kwargs = {'input': x, 'dim': 1, 'dtype': torch.float64}
        result = torch.log_softmax(**kwargs)
        """
    )
    obj.run(pytorch_code, ["result"])
