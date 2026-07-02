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

import paddle
import pytest
from apibase import APIBase

obj = APIBase("torch.nn.functional.rms_norm")

rms_norm_not_supported = (
    not paddle.device.is_compiled_with_cuda()
    and not paddle.device.is_compiled_with_xpu()
)


@pytest.mark.skipif(
    rms_norm_not_supported, reason="rms_norm kernel is only registered on GPU/XPU"
)
def test_case_1():
    """Positional arguments test"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = F.rms_norm(x, (3,))
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skipif(
    rms_norm_not_supported, reason="rms_norm kernel is only registered on GPU/XPU"
)
def test_case_2():
    """Keyword arguments test"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = F.rms_norm(input=x, normalized_shape=(3,))
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skipif(
    rms_norm_not_supported, reason="rms_norm kernel is only registered on GPU/XPU"
)
def test_case_3():
    """Keyword arguments with weight and eps"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        weight = torch.ones(3)
        result = F.rms_norm(x, (3,), weight=weight, eps=1e-5)
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skipif(
    rms_norm_not_supported, reason="rms_norm kernel is only registered on GPU/XPU"
)
def test_case_4():
    """Mixed arguments test"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        weight = torch.ones(3)
        result = F.rms_norm(x, (3,), weight, 1e-5)
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skipif(
    rms_norm_not_supported, reason="rms_norm kernel is only registered on GPU/XPU"
)
def test_case_5():
    """Keyword arguments out of order"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        weight = torch.ones(3)
        result = F.rms_norm(eps=1e-5, weight=weight, normalized_shape=(3,), input=x)
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skipif(
    rms_norm_not_supported, reason="rms_norm kernel is only registered on GPU/XPU"
)
def test_case_6():
    """3D input tensor test"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
        result = F.rms_norm(x, (3,))
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skipif(
    rms_norm_not_supported, reason="rms_norm kernel is only registered on GPU/XPU"
)
def test_case_7():
    """With weight and default eps"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        weight = torch.full((3,), 0.5)
        result = F.rms_norm(x, (3,), weight=weight)
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skipif(
    rms_norm_not_supported, reason="rms_norm kernel is only registered on GPU/XPU"
)
def test_case_8():
    """Higher dimensional input"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
        result = F.rms_norm(x, (4,))
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skipif(
    rms_norm_not_supported, reason="rms_norm kernel is only registered on GPU/XPU"
)
def test_case_9():
    """Gradient computation test"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[0.5, 1.0, 2.0], [3.0, 4.0, 5.0]], requires_grad=True)
        y = torch.nn.functional.rms_norm(a, (3,))
        y.sum().backward()
        a_grad = a.grad
        """
    )
    obj.run(pytorch_code, ["y", "a_grad"], check_stop_gradient=False, check_value=False)


@pytest.mark.skipif(
    rms_norm_not_supported, reason="rms_norm kernel is only registered on GPU/XPU"
)
def test_case_10():
    """Expression argument test"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.nn.functional.rms_norm(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), (3,))
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skipif(
    rms_norm_not_supported, reason="rms_norm kernel is only registered on GPU/XPU"
)
def test_case_11():
    """Expression argument test"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.nn.functional.rms_norm(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), (3,))
        """
    )
    obj.run(pytorch_code, ["result"])
