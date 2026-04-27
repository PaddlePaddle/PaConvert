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

obj = APIBase("torch.nn.functional.log_softmax")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.tensor([[[-2.0, 3.0, -4.0, 5.0],
                            [3.0, -4.0, 5.0, -6.0],
                            [-7.0, -8.0, 8.0, 9.0]],
                            [[1.0, -2.0, -3.0, 4.0],
                            [-5.0, 6.0, 7.0, -8.0],
                            [6.0, 7.0, 8.0, 9.0]]])
        result = F.log_softmax(x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.tensor([[[-2.0, 3.0, -4.0, 5.0],
                            [3.0, -4.0, 5.0, -6.0],
                            [-7.0, -8.0, 8.0, 9.0]],
                            [[1.0, -2.0, -3.0, 4.0],
                            [-5.0, 6.0, 7.0, -8.0],
                            [6.0, 7.0, 8.0, 9.0]]])
        result = F.log_softmax(x, dim=1)
        """
    )
    obj.run(pytorch_code, ["result"], atol=1e-05, rtol=1e-06)


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.tensor([[[-2.0, 3.0, -4.0, 5.0],
                            [3.0, -4.0, 5.0, -6.0],
                            [-7.0, -8.0, 8.0, 9.0]],
                            [[1.0, -2.0, -3.0, 4.0],
                            [-5.0, 6.0, 7.0, -8.0],
                            [6.0, 7.0, 8.0, 9.0]]])
        result = F.log_softmax(x, dtype=torch.float64)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.tensor([[[-2.0, 3.0, -4.0, 5.0],
                            [3.0, -4.0, 5.0, -6.0],
                            [-7.0, -8.0, 8.0, 9.0]],
                            [[1.0, -2.0, -3.0, 4.0],
                            [-5.0, 6.0, 7.0, -8.0],
                            [6.0, 7.0, 8.0, 9.0]]])
        result = F.log_softmax(x, _stacklevel=2)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.tensor([[[-2.0, 3.0, -4.0, 5.0],
                            [3.0, -4.0, 5.0, -6.0],
                            [-7.0, -8.0, 8.0, 9.0]],
                            [[1.0, -2.0, -3.0, 4.0],
                            [-5.0, 6.0, 7.0, -8.0],
                            [6.0, 7.0, 8.0, 9.0]]])
        result = F.log_softmax(input=x, dim=2, _stacklevel=2, dtype=torch.float64)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.tensor([[[-2.0, 3.0, -4.0, 5.0],
                            [3.0, -4.0, 5.0, -6.0],
                            [-7.0, -8.0, 8.0, 9.0]],
                            [[1.0, -2.0, -3.0, 4.0],
                            [-5.0, 6.0, 7.0, -8.0],
                            [6.0, 7.0, 8.0, 9.0]]])
        result = F.log_softmax(x, 2, 2, torch.float64)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.tensor([[[-2.0, 3.0, -4.0, 5.0],
                            [3.0, -4.0, 5.0, -6.0],
                            [-7.0, -8.0, 8.0, 9.0]],
                            [[1.0, -2.0, -3.0, 4.0],
                            [-5.0, 6.0, 7.0, -8.0],
                            [6.0, 7.0, 8.0, 9.0]]])
        result = F.log_softmax(input=x, _stacklevel=2, dtype=torch.float64, dim=-2)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    # 1D input, dim=0 positional
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = F.log_softmax(x, 0)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    # 1D input, dim=0 keyword
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = F.log_softmax(x, dim=0)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_10():
    # 2D input, dim=0
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = F.log_softmax(x, dim=0)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_11():
    # 2D input, dim=1 (default for 2D when dim=None would be 1)
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = F.log_softmax(x, dim=1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_12():
    # 2D input, dim=-1
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = F.log_softmax(x, dim=-1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_13():
    # float64 input tensor
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64)
        result = F.log_softmax(x, dim=1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_14():
    # float64 input with dtype specified (no-op cast)
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64)
        result = F.log_softmax(x, dim=1, dtype=torch.float64)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_15():
    # 4D input, dim=3
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]])
        result = F.log_softmax(x, dim=3)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_16():
    # gradient computation
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
        result = F.log_softmax(x, dim=1)
        result.sum().backward()
        x_grad = x.grad
        """
    )
    obj.run(pytorch_code, ["result", "x_grad"], check_stop_gradient=False)


def test_case_17():
    # variable args
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        args = (x, 1)
        result = F.log_softmax(*args)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_18():
    # input keyword alias
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = F.log_softmax(input=x, dim=0)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_19():
    # out-of-order keyword args: dtype before dim
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = F.log_softmax(dtype=torch.float64, dim=1, input=x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_20():
    # dim=-2 on 3D input
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        result = F.log_softmax(x, dim=-2)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_21():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = F.log_softmax(x, 1, 3, dtype=torch.float64)
        """
    )
    obj.run(pytorch_code, ["result"])
