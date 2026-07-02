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

obj = APIBase("torch.cuda.CUDAGraph")


def test_case_1():
    """Conversion check: torch.cuda.CUDAGraph maps to paddle.cuda.CUDAGraph"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        g = torch.cuda.CUDAGraph()
        result = isinstance(g, torch.cuda.CUDAGraph)
        """
    )
    obj.run(
        pytorch_code,
        expect_paddle_code="import paddle\n\ng = paddle.cuda.CUDAGraph()\nresult = isinstance(g, paddle.cuda.CUDAGraph)\n",
    )


@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="can only run on paddle with CUDA",
)
def test_case_2():
    """Basic graph capture and replay"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.full([4], 3.0, device="cuda")
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        torch.cuda.synchronize()
        with torch.cuda.stream(s):
            g.capture_begin()
            y = x * 2.0 + 1.0
            g.capture_end()
        torch.cuda.synchronize()
        g.replay()
        torch.cuda.synchronize()
        result = y
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="can only run on paddle with CUDA",
)
def test_case_3():
    """Replay recomputes from the updated static input tensor"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.full([4], 1.0, device="cuda")
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        torch.cuda.synchronize()
        with torch.cuda.stream(s):
            g.capture_begin()
            y = x + 5.0
            g.capture_end()
        torch.cuda.synchronize()
        g.replay()
        torch.cuda.synchronize()
        result1 = y.clone()
        x.copy_(torch.full([4], 10.0, device="cuda"))
        g.replay()
        torch.cuda.synchronize()
        result2 = y.clone()
        """
    )
    obj.run(pytorch_code, ["result1", "result2"])


@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="can only run on paddle with CUDA",
)
def test_case_4():
    """Multiple replays accumulate on a static buffer captured in-place"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        counter = torch.zeros([3], device="cuda")
        one = torch.ones([3], device="cuda")
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        torch.cuda.synchronize()
        with torch.cuda.stream(s):
            g.capture_begin()
            counter.add_(one)
            g.capture_end()
        torch.cuda.synchronize()
        g.replay()
        g.replay()
        g.replay()
        torch.cuda.synchronize()
        result = counter
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="can only run on paddle with CUDA",
)
def test_case_5():
    """Graph can be reset after replay"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.full([2, 2], 4.0, device="cuda")
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        torch.cuda.synchronize()
        with torch.cuda.stream(s):
            g.capture_begin()
            y = x * x
            g.capture_end()
        torch.cuda.synchronize()
        g.replay()
        torch.cuda.synchronize()
        result = y.clone()
        g.reset()
        """
    )
    obj.run(pytorch_code, ["result"])
