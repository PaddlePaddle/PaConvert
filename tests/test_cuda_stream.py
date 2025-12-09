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

obj = APIBase("torch.cuda.Stream")


@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="can only run on paddle with CUDA",
)
def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        stream = torch.cuda.Stream()
        result = stream.query()
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
        stream = torch.cuda.Stream(priority=0)
        result = stream.query()
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
        stream = torch.cuda.Stream(priority=-1)
        result = stream.query()
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
        stream = torch.cuda.Stream(device=1)
        result = stream.query()
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
        stream = torch.cuda.Stream(device=1,priority=-1)
        result = stream.query()
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
        stream = torch.cuda.Stream(device='cuda:1',priority=-1)
        result = stream.query()
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
        stream = torch.cuda.Stream(device='cuda',priority=-1)
        result = stream.query()
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
        stream = torch.cuda.Stream(0,-1)
        result = stream.query()
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
        stream = torch.cuda.Stream('cuda:0',-1)
        result = stream.query()
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
        cond = True
        stream = torch.cuda.Stream(0 if cond else 1, -1)
        result = stream.query()
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
        cond = False
        stream = torch.cuda.Stream('cuda:0' if cond else 'cuda:1', -1)
        result = stream.query()
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
        device = 0
        stream = torch.cuda.Stream(device,-1)
        result = stream.query()
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
        device = 'cuda:0'
        stream = torch.cuda.Stream(device,-1)
        result = stream.query()
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
        cond = True
        device = 0 if cond else 1
        stream = torch.cuda.Stream(device, -1)
        result = stream.query()
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
        cond = False
        device = 'cuda:0' if cond else 'cuda:1'
        stream = torch.cuda.Stream(device, -1)
        result = stream.query()
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
        stream = torch.cuda.Stream(torch.device(0), -1)
        result = stream.query()
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
        stream = torch.cuda.Stream(torch.device('cuda:0'), -1)
        result = stream.query()
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
        cond = True
        stream = torch.cuda.Stream(torch.device(0 if cond else 1), -1)
        result = stream.query()
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
        cond = False
        stream = torch.cuda.Stream(torch.device('cuda:0' if cond else 'cuda:1'), -1)
        result = stream.query()
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
        device = 0
        stream = torch.cuda.Stream(torch.device(device), -1)
        result = stream.query()
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
        device = 'cuda:0'
        stream = torch.cuda.Stream(torch.device(device), -1)
        result = stream.query()
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
        cond = True
        device = 0 if cond else 1
        stream = torch.cuda.Stream(torch.device(device), -1)
        result = stream.query()
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="can only run on paddle with CUDA",
)
def test_case_23():
    pytorch_code = textwrap.dedent(
        """
        import torch
        cond = False
        device = 'cuda:0' if cond else 'cuda:1'
        stream = torch.cuda.Stream(torch.device(device), -1)
        result = stream.query()
        """
    )
    obj.run(pytorch_code, ["result"])
