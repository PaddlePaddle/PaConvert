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

from apibase import APIBase

obj = APIBase("torch.Tensor.isclose")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.tensor([10000., 1e-07]).isclose(torch.tensor([10000.1, 1e-08]))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.tensor([10000., 1e-08]).isclose(torch.tensor([10000.1, 1e-09]))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.tensor([1.0, float('nan')]).isclose(torch.tensor([1.0, float('nan')]))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.tensor([1.0, float('inf')]).isclose(torch.tensor([1.0, float('inf')]), equal_nan=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.tensor([10000., 1e-07]).isclose(torch.tensor([10000.1, 1e-08]), atol=2.)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.tensor([10000., 1e-07]).isclose(other=torch.tensor([10000.1, 1e-08]), rtol=1e-5,  atol=2., equal_nan=False)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.tensor([10000., 1e-07]).isclose(torch.tensor([10000.1, 1e-08]), 1e-5, 2., False)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.tensor([10000., 1e-07]).isclose(equal_nan=False, other=torch.tensor([10000.1, 1e-08]), rtol=1e-5,  atol=2.)
        """
    )
    obj.run(pytorch_code, ["result"])
