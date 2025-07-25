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

import numpy as np
from apibase import APIBase


class TensorToSpareCsrAPIBase(APIBase):
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
        pytorch_numpy = pytorch_result.cpu().to_dense().numpy()
        paddle_numpy = paddle_result.to_dense().numpy()
        assert (
            pytorch_numpy.dtype == paddle_numpy.dtype
        ), "API ({}): dtype mismatch, torch dtype is {}, paddle dtype is {}".format(
            name, pytorch_numpy.dtype, paddle_numpy.dtype
        )
        assert (
            pytorch_numpy.shape == paddle_numpy.shape
        ), "API ({}): shape mismatch, torch shape is {}, paddle shape is {}".format(
            name, pytorch_numpy.shape, paddle_numpy.shape
        )
        np.testing.assert_allclose(
            pytorch_numpy, paddle_numpy, rtol=rtol, atol=atol
        ), "API ({}): paddle result has diff with pytorch result".format(name)


obj = TensorToSpareCsrAPIBase("torch.Tensor.to_sparse_csr")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        dense_tensor = torch.tensor([
            [10, 0, 0],
            [0, 0, 20],
            [0, 30, 40]
        ])
        result = dense_tensor.to_sparse_csr()
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
    )


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        dense_tensor = torch.tensor([
            [[1, 2], [0, 0], [0, 0]],
            [[0, 0], [0, 0], [3, 4]],
            [[0, 0], [5, 6], [7, 8]]
        ])
        result = dense_tensor.to_sparse_csr(dense_dim=1)
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        unsupport=True,
        reason="not support parameter: 'dense_dim'",
    )


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        dense_tensor = torch.tensor([[1, 0, 2, 0]])
        result = dense_tensor.to_sparse_csr()
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
    )
