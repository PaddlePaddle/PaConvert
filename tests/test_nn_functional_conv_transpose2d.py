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

obj = APIBase("torch.nn.functional.conv_transpose2d")


def test_case_1():
    """basic usage: input + weight only"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        weight = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
        result = F.conv_transpose2d(x, weight)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    """with bias as positional argument"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        weight = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
        bias = torch.tensor([0.5])
        result = F.conv_transpose2d(x, weight, bias)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    """stride as keyword argument"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        weight = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
        result = F.conv_transpose2d(x, weight, stride=2)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    """stride + padding as keyword arguments"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        weight = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
        result = F.conv_transpose2d(x, weight, stride=2, padding=1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    """stride + padding + dilation"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        weight = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
        result = F.conv_transpose2d(x, weight, stride=2, padding=1, dilation=1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    """groups parameter"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.ones(1, 4, 3, 3)
        weight = torch.ones(4, 1, 2, 2)
        result = F.conv_transpose2d(x, weight, stride=1, padding=0, dilation=1, groups=2)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    """output_padding with stride > 1"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.ones(1, 4, 3, 3)
        weight = torch.ones(4, 1, 2, 2)
        result = F.conv_transpose2d(x, weight, stride=2, padding=1, output_padding=1, dilation=1, groups=2)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    """all keyword arguments with bias=None"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.ones(1, 4, 3, 3)
        weight = torch.ones(4, 1, 2, 2)
        result = F.conv_transpose2d(x, weight, bias=None, stride=2, padding=1, output_padding=1, dilation=1, groups=2)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    """all positional arguments"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.ones(1, 4, 3, 3)
        weight = torch.ones(4, 1, 2, 2)
        result = F.conv_transpose2d(x, weight, None, 2, 1, 1, 2, 1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_10():
    """all keyword arguments in shuffled order (groups before dilation)"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.ones(1, 4, 3, 3)
        weight = torch.ones(4, 1, 2, 2)
        result = F.conv_transpose2d(input=x, weight=weight, bias=None, stride=2, padding=1, output_padding=1, groups=2, dilation=1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_11():
    """bias as keyword argument"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        weight = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
        bias = torch.tensor([1.0])
        result = F.conv_transpose2d(x, weight, bias=bias)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_12():
    """tuple stride parameter"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        weight = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
        result = F.conv_transpose2d(x, weight, stride=(2, 1))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_13():
    """tuple padding parameter"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]])
        weight = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
        result = F.conv_transpose2d(x, weight, stride=2, padding=(1, 0))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_14():
    """tuple dilation parameter"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        weight = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
        result = F.conv_transpose2d(x, weight, dilation=(1, 2))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_15():
    """tuple output_padding parameter"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        weight = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
        result = F.conv_transpose2d(x, weight, stride=(2, 3), output_padding=(1, 2))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_16():
    """all tuple parameters combined"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.ones(1, 2, 4, 4)
        weight = torch.ones(2, 1, 3, 3)
        result = F.conv_transpose2d(x, weight, stride=(2, 2), padding=(1, 1), output_padding=(1, 1), dilation=(1, 1), groups=2)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_17():
    """float64 dtype"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=torch.float64)
        weight = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=torch.float64)
        result = F.conv_transpose2d(x, weight)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_18():
    """multiple input and output channels"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.ones(1, 2, 3, 3)
        weight = torch.ones(2, 3, 2, 2)
        bias = torch.tensor([0.1, 0.2, 0.3])
        result = F.conv_transpose2d(x, weight, bias)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_19():
    """keyword arguments fully reordered"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.ones(1, 2, 3, 3)
        weight = torch.ones(2, 3, 2, 2)
        bias = torch.tensor([0.1, 0.2, 0.3])
        result = F.conv_transpose2d(dilation=1, groups=1, output_padding=0, padding=0, stride=1, bias=bias, weight=weight, input=x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_20():
    """mixed positional and keyword arguments"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.ones(1, 2, 3, 3)
        weight = torch.ones(2, 3, 2, 2)
        result = F.conv_transpose2d(x, weight, None, stride=2, padding=1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_21():
    """batch size of 1 with larger spatial dimensions"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.ones(1, 1, 5, 5)
        weight = torch.ones(1, 1, 3, 3)
        result = F.conv_transpose2d(x, weight)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_22():
    """non-square kernel"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.ones(1, 1, 4, 4)
        weight = torch.ones(1, 1, 2, 3)
        result = F.conv_transpose2d(x, weight)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_23():
    """gradient computation"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], requires_grad=True)
        weight = torch.tensor([[[[1.0, 0.5], [0.5, 1.0]]]], requires_grad=True)
        y = F.conv_transpose2d(x, weight)
        y.sum().backward()
        result = y
        x_grad = x.grad
        """
    )
    obj.run(pytorch_code, ["result", "x_grad"], check_stop_gradient=False)


def test_case_24():
    """expression as argument"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.ones(1, 1, 4, 4)
        weight = torch.ones(1, 1, 2, 2)
        result = F.conv_transpose2d(x, weight, stride=1 + 1, padding=2 - 1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_25():
    """variable arguments via unpacking"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.ones(1, 1, 4, 4)
        weight = torch.ones(1, 1, 2, 2)
        kwargs = dict(stride=2, padding=1, output_padding=1)
        result = F.conv_transpose2d(x, weight, **kwargs)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_26():
    """multiple batches"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.ones(4, 1, 3, 3)
        weight = torch.ones(1, 2, 2, 2)
        bias = torch.tensor([0.5, -0.5])
        result = F.conv_transpose2d(x, weight, bias, stride=1, padding=0)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_27():
    """asymmetric stride and padding as tuples"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.ones(1, 1, 4, 4)
        weight = torch.ones(1, 1, 3, 3)
        result = F.conv_transpose2d(x, weight, stride=(1, 2), padding=(0, 1))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_28():
    """default parameters explicitly specified"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        weight = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
        result = F.conv_transpose2d(x, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_29():
    """default parameters omitted (same as test_case_28 for value comparison)"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        weight = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
        result = F.conv_transpose2d(x, weight)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_30():
    """larger groups with bias"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.ones(1, 4, 3, 3)
        weight = torch.ones(4, 1, 2, 2)
        bias = torch.tensor([0.1, 0.2, 0.3, 0.4])
        result = F.conv_transpose2d(x, weight, bias, groups=4)
        """
    )
    obj.run(pytorch_code, ["result"])
