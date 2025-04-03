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

obj = APIBase("torch.Tensor.cuda")


@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="can only run on paddle with CUDA",
)
def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([1,2,3])
        result = a.cuda()
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="can only run on paddle with CUDA",
)
def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.zeros((1,2,3,4))
        result = a.cuda(device="cuda:0", non_blocking=True, memory_format=torch.channels_last)
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="can only run on paddle with CUDA",
)
def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.zeros((1,2,3,4))
        result = a.cuda("cuda:0", True)
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
        a = torch.zeros((1,2,3,4))
        result = a.cuda(non_blocking=True, device="cuda:0", memory_format=torch.channels_last)
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="can only run on paddle with CUDA",
)
def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.zeros((1,2,3,4))
        result = a.cuda(0)
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="can only run on paddle with CUDA",
)
def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.zeros((1,2,3,4))
        result = a.cuda('cuda:0')
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="can only run on paddle with CUDA",
)
def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.zeros((1,2,3,4))
        result = a.cuda(0 if 1 > 0 else 1)
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="can only run on paddle with CUDA",
)
def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.zeros((1,2,3,4))
        result = a.cuda("cuda:0" if 1 > 0 else "cuda:1")
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="can only run on paddle with CUDA",
)
def test_case_9():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.zeros((1,2,3,4))
        device = 0
        result = a.cuda(device)
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="can only run on paddle with CUDA",
)
def test_case_10():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.zeros((1,2,3,4))
        device = "cuda:0"
        result = a.cuda(device)
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="can only run on paddle with CUDA",
)
def test_case_11():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.zeros((1,2,3,4))
        device = 0 if 1 > 0 else 1
        result = a.cuda(device)
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="can only run on paddle with CUDA",
)
def test_case_12():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.zeros((1,2,3,4))
        device = "cuda:0" if 1 > 0 else "cuda:1"
        result = a.cuda(device)
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="can only run on paddle with CUDA",
)
def test_case_13():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.zeros((1,2,3,4))
        result = a.cuda(device = 0 if 1 > 0 else 1)
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="can only run on paddle with CUDA",
)
def test_case_14():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.zeros((1,2,3,4))
        result = a.cuda(device = "cuda:0" if 1 > 0 else "cuda:1")
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="can only run on paddle with CUDA",
)
def test_case_15():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.zeros((1,2,3,4))
        device = 0 if 1 > 0 else 1
        result = a.cuda(device=device)
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="can only run on paddle with CUDA",
)
def test_case_16():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.zeros((1,2,3,4))
        device = "cuda:0" if 1 > 0 else "cuda:1"
        result = a.cuda(device=device)
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="can only run on paddle with CUDA",
)
def test_case_17():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.zeros((1,2,3,4))
        device = torch.device(0 if 1 > 0 else 1)
        result = a.cuda(device)
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="can only run on paddle with CUDA",
)
def test_case_18():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.zeros((1,2,3,4))
        device = torch.device("cuda:0" if 1 > 0 else "cuda:1")
        result = a.cuda(device)
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="can only run on paddle with CUDA",
)
def test_case_19():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.zeros((1,2,3,4))
        result = a.cuda(device = torch.device(0 if 1 > 0 else 1))
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="can only run on paddle with CUDA",
)
def test_case_20():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.zeros((1,2,3,4))
        result = a.cuda(device = torch.device("cuda:0" if 1 > 0 else "cuda:1"))
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="can only run on paddle with CUDA",
)
def test_case_21():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.zeros((1,2,3,4))
        device = torch.device(0 if 1 > 0 else 1)
        result = a.cuda(device=device)
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="can only run on paddle with CUDA",
)
def test_case_22():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.zeros((1,2,3,4))
        device = torch.device("cuda:0" if 1 > 0 else "cuda:1")
        result = a.cuda(device=device)
        """
    )
    obj.run(pytorch_code, ["result"])
