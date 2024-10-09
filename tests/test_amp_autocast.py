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
#
import textwrap

import numpy as np
import paddle
import pytest
from apibase import APIBase


class AmpAutocastBase(APIBase):
    def compare(
        self,
        name,
        pytorch_result,
        paddle_result,
        check_value=True,
        check_dtype=True,
        check_stop_gradient=True,
        rtol=1.0e-6,
        atol=0.0,
    ):
        (
            pytorch_numpy,
            paddle_numpy,
        ) = pytorch_result.float().cpu().detach().numpy(), paddle_result.astype(
            "float32"
        ).numpy(
            False
        )
        assert (
            pytorch_numpy.shape == paddle_numpy.shape
        ), "API ({}): shape mismatch, torch shape is {}, paddle shape is {}".format(
            name, pytorch_numpy.shape, paddle_numpy.shape
        )
        assert (
            pytorch_numpy.dtype == paddle_numpy.dtype
        ), "API ({}): dtype mismatch, torch dtype is {}, paddle dtype is {}".format(
            name, pytorch_numpy.dtype, paddle_numpy.dtype
        )
        if check_value:
            assert np.allclose(
                pytorch_numpy, paddle_numpy, rtol=rtol, atol=atol
            ), "API ({}): paddle result has diff with pytorch result".format(name)


obj = AmpAutocastBase("torch.amp.autocast")


@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="can only run on paddle with CUDA",
)
def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch

        model = torch.nn.Linear(10, 5, device="cuda")
        input = torch.randn(4, 10, device="cuda")

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=False, cache_enabled=True):
            result = model(input)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="can only run on paddle with CUDA",
)
def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch

        model = torch.nn.Linear(10, 5, device="cuda")
        input = torch.randn(4, 10, device="cuda")

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=False, cache_enabled=True):
            result = model(input)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="can only run on paddle with CUDA",
)
def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch

        model = torch.nn.Linear(10, 5, device="cuda")
        input = torch.randn(4, 10, device="cuda")

        with torch.amp.autocast(device_type="cuda"):
            result = model(input)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


@pytest.mark.skipif(
    condition=not paddle.device.is_compiled_with_cuda(),
    reason="can only run on paddle with CUDA",
)
def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch

        model = torch.nn.Linear(10, 5, device="cuda")
        input = torch.randn(4, 10, device="cuda")

        with torch.amp.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda", cache_enabled=True):
            result = model(input)
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

        model = torch.nn.Linear(10, 5, device="cuda")
        input = torch.randn(4, 10, device="cuda")

        with torch.amp.autocast("cuda", torch.float16, False, True):
            result = model(input)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)
