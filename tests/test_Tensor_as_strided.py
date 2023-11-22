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

from apibase import APIBase

obj = APIBase("torch.Tensor.as_strided")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[ 0.0335,  0.1830, -0.1269],
        [ 0.1897, -0.1422, -0.4940],
        [-0.7674, -0.0134, -0.3733]])
        results = x.as_strided((2, 2), (1, 2))
        """
    )
    obj.run(pytorch_code, ["results"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[ 0.0335,  0.1830, -0.1269],
        [ 0.1897, -0.1422, -0.4940],
        [-0.7674, -0.0134, -0.3733]])
        results = x.as_strided((2, 2), (1, 2), 0)
        """
    )
    obj.run(pytorch_code, ["results"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[ 0.0335,  0.1830, -0.1269],
        [ 0.1897, -0.1422, -0.4940],
        [-0.7674, -0.0134, -0.3733]])
        size = (2, 2)
        stride = (1, 2)
        storage_offset = 0
        results = x.as_strided(size, stride, storage_offset)
        """
    )
    obj.run(pytorch_code, ["results"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[ 0.0335,  0.1830, -0.1269],
        [ 0.1897, -0.1422, -0.4940],
        [-0.7674, -0.0134, -0.3733]])
        size = (2, 2)
        stride = (1, 2)
        results = x.as_strided(size, stride, 0)
        """
    )
    obj.run(pytorch_code, ["results"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[ 0.0335,  0.1830, -0.1269],
        [ 0.1897, -0.1422, -0.4940],
        [-0.7674, -0.0134, -0.3733]])
        size = (2, 2)
        stride = (1, 2)
        results = x.as_strided(size = (2,2), stride = (2,2), storage_offset = 0)
        """
    )
    obj.run(pytorch_code, ["results"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[ 0.0335,  0.1830, -0.1269],
        [ 0.1897, -0.1422, -0.4940],
        [-0.7674, -0.0134, -0.3733]])
        size = (2, 2)
        stride = (1, 2)
        results = x.as_strided(stride = (2,2), size = (2,2), storage_offset = 0)
        """
    )
    obj.run(pytorch_code, ["results"])
