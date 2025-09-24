# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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


class DeviceAPIBase(APIBase):
    def compare(
        self,
        name,
        pytorch_result,
        paddle_result,
        check_value=True,
        check_shape=True,
        check_dtype=True,
        check_stop_gradient=True,
        rtol=1.0e-6,
        atol=0.0,
    ):
        pytorch_result = str(pytorch_result).replace("cuda", "gpu")

        if "cpu:" in pytorch_result:
            pytorch_result = "cpu"
        assert pytorch_result == paddle_result


obj = DeviceAPIBase("torch.device")

obj = DeviceAPIBase("torch.set_default_device")


@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="can only run on paddle with CUDA",
)
def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        torch.set_default_device('cuda:1')
        result = torch.get_default_device()
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
        torch.set_default_device("cuda")
        result = torch.get_default_device()
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        torch.set_default_device("cpu")
        result = torch.get_default_device()

        # if not set None, will cause test_vander error
        torch.set_default_device(None)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        torch.set_default_device("cpu:1")
        result = torch.get_default_device()

        # if not set None, will cause test_vander error
        torch.set_default_device(None)
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
        torch.set_default_device(device=torch.device("cuda"))
        result = torch.get_default_device()
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
        torch.set_default_device(device=torch.device("cuda:1"))
        result = torch.get_default_device()
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
        device = torch.device("cuda")
        torch.set_default_device(device)
        result = torch.get_default_device()
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        device = torch.device("cpu:1")
        torch.set_default_device(device)
        result = torch.get_default_device()

        # if not set None, will cause test_vander error
        torch.set_default_device(None)
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
        device = "cuda:0"
        torch.set_default_device(device=device)
        result = torch.get_default_device()
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_10():
    pytorch_code = textwrap.dedent(
        """
        import torch
        device = 'cpu'
        torch.set_default_device(device=device)
        result = torch.get_default_device()

        # if not set None, will cause test_vander error
        torch.set_default_device(None)
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
        cond = True
        torch.set_default_device(device='cuda' if cond else 'cpu')
        result = torch.get_default_device()
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
        cond = True
        torch.set_default_device(device='cuda:0' if cond else 'cuda:1')
        result = torch.get_default_device()
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
        cond = False
        device = 'cuda' if cond else 'cpu'
        torch.set_default_device(device=device)
        result = torch.get_default_device()

        # if not set None, will cause test_vander error
        torch.set_default_device(None)
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
        cond = False
        device = "cuda:0" if cond else "cuda:1"
        torch.set_default_device(device=device)
        result = torch.get_default_device()
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
        torch.set_default_device(0)
        result = torch.get_default_device()
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
        cond = True
        torch.set_default_device(device=0 if False else 1)
        result = torch.get_default_device()
        """
    )
    obj.run(pytorch_code, ["result"])
