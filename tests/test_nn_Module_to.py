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

obj = APIBase("torch.nn.Module.to")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1., 2., 3.])
        module1 = torch.nn.Module()
        module1.register_buffer('buffer', x)
        module1.to(dtype=torch.float32)
        result = module1.buffer
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1., 2., 3.])
        module1 = torch.nn.Module()
        module1.register_buffer('buffer', x)
        module1.to(device="cpu")
        result = module1.buffer
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1., 2., 3.])
        module1 = torch.nn.Module()
        module1.register_buffer('buffer', x)
        module1.to(device="cpu", non_blocking=False)
        result = module1.buffer
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="can only run on paddle with CUDA",
)
def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        linear = torch.nn.Linear(10, 10)
        linear.to("cuda", non_blocking=False)
        result = linear.weight
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="can only run on paddle with CUDA",
)
def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        linear = torch.nn.Linear(10, 10)
        linear.to(device="cuda:0", non_blocking=False)
        result = linear.weight
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="can only run on paddle with CUDA",
)
def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        linear = torch.nn.Linear(10, 10)
        linear.to(torch.device('cuda'), non_blocking=False)
        result = linear.weight
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="can only run on paddle with CUDA",
)
def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        linear = torch.nn.Linear(10, 10)
        linear.to(device=torch.device('cuda:0'), non_blocking=False)
        result = linear.bias
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_8():
    """Positional dtype: to(torch.float64)"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        module = torch.nn.Module()
        module.register_buffer('buf', torch.tensor([1.0, 2.0, 3.0]))
        module.to(torch.float64)
        result = module.buf
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    """Positional dtype: to(torch.float16)"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        module = torch.nn.Module()
        module.register_buffer('buf', torch.tensor([1.0, 2.0, 3.0]))
        module.to(torch.float16)
        result = module.buf
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_10():
    """Keyword dtype: to(dtype=torch.float64)"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        module = torch.nn.Module()
        module.register_buffer('buf', torch.tensor([1.0, 2.0, 3.0]))
        module.to(dtype=torch.float64)
        result = module.buf
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_11():
    """Positional device string: to('cpu')"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        module = torch.nn.Module()
        module.register_buffer('buf', torch.tensor([1.0, 2.0, 3.0]))
        module.to('cpu')
        result = module.buf
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_12():
    """Positional device + keyword dtype: to('cpu', dtype=torch.float64)"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        module = torch.nn.Module()
        module.register_buffer('buf', torch.tensor([1.0, 2.0, 3.0]))
        module.to('cpu', dtype=torch.float64)
        result = module.buf
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_13():
    """Positional tensor: to(some_tensor)"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        module = torch.nn.Module()
        module.register_buffer('buf', torch.tensor([1.0, 2.0, 3.0]))
        ref = torch.tensor([1.0], dtype=torch.float64)
        module.to(ref)
        result = module.buf
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_14():
    """Chaining: ret = module.to(torch.float64)"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        module = torch.nn.Module()
        module.register_buffer('buf', torch.tensor([1.0, 2.0, 3.0]))
        result = module.to(torch.float64).buf
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_15():
    """floating_only: int buffer should NOT be cast"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        module = torch.nn.Module()
        module.register_buffer('float_buf', torch.tensor([1.0, 2.0, 3.0]))
        module.register_buffer('int_buf', torch.tensor([1, 2, 3]))
        module.to(torch.float64)
        result = module.int_buf
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_16():
    """floating_only: float buffer should be cast"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        module = torch.nn.Module()
        module.register_buffer('float_buf', torch.tensor([1.0, 2.0, 3.0]))
        module.register_buffer('int_buf', torch.tensor([1, 2, 3]))
        module.to(torch.float64)
        result = module.float_buf
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_17():
    """Keyword non_blocking with positional dtype"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        module = torch.nn.Module()
        module.register_buffer('buf', torch.tensor([1.0, 2.0, 3.0]))
        module.to(torch.float64, non_blocking=False)
        result = module.buf
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_18():
    """Keyword device and dtype together"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        module = torch.nn.Module()
        module.register_buffer('buf', torch.tensor([1.0, 2.0, 3.0]))
        module.to(device='cpu', dtype=torch.float64)
        result = module.buf
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_19():
    """Reordered kwargs: dtype first, device second"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        module = torch.nn.Module()
        module.register_buffer('buf', torch.tensor([1.0, 2.0, 3.0]))
        module.to(dtype=torch.float64, device='cpu')
        result = module.buf
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_20():
    """Sequential to() calls"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        module = torch.nn.Module()
        module.register_buffer('buf', torch.tensor([1.0, 2.0, 3.0]))
        module.to(torch.float64)
        module.to(torch.float32)
        result = module.buf
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_21():
    """Sublayers should be cast too"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        module = torch.nn.Module()
        module.register_buffer('buf', torch.tensor([1.0, 2.0, 3.0]))
        sub = torch.nn.Module()
        sub.register_buffer('sub_buf', torch.tensor([4.0, 5.0, 6.0]))
        module.add_module('sub', sub)
        module.to(torch.float64)
        result = module.sub.sub_buf
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_22():
    """to() with no args returns self"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        module = torch.nn.Module()
        module.register_buffer('buf', torch.tensor([1.0, 2.0, 3.0]))
        result = module.to().buf
        """
    )
    obj.run(pytorch_code, ["result"])
