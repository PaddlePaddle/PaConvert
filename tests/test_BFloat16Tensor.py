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

import numpy as np
from apibase import APIBase


class BFloat16TensorAPIBase(APIBase):
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
        ) = pytorch_result.float().cpu().numpy(), paddle_result.astype("float32").numpy(
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


obj = BFloat16TensorAPIBase("torch.BFloat16Tensor")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.BFloat16Tensor([1.5, 2, 3])
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.BFloat16Tensor()
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.BFloat16Tensor(2)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.BFloat16Tensor(2,3)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        shape = [3, 5]
        result = torch.BFloat16Tensor(*shape)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        def fun(x: torch.BFloat16Tensor):
            return x * 2

        a = torch.BFloat16Tensor(5, 4)
        result = fun(a)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.BFloat16Tensor((3, 2, 3))
        """
    )
    obj.run(pytorch_code, ["result"])
