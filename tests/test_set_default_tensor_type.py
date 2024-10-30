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

obj = APIBase("torch.set_default_tensor_type")


# These test has been run locally. Due to torch.set_default_tensor_type would make a global setting,
# other tests are affected. So we disable most of them here.
def _test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        result = torch.tensor([1.2, 3])
        """
    )
    obj.run(pytorch_code, ["result"])


def _test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        torch.set_default_tensor_type(t=torch.cuda.HalfTensor)
        result = torch.tensor([1.2, 3])
        """
    )
    obj.run(pytorch_code, ["result"])


def _test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        torch.set_default_tensor_type(t="torch.cuda.HalfTensor")
        result = torch.tensor([1.2, 3.8])
        """
    )
    obj.run(pytorch_code, ["result"])


def _test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        torch.set_default_tensor_type("torch.DoubleTensor")
        result = torch.tensor([1.2, 3.8])
        """
    )
    obj.run(pytorch_code, ["result"])


def _test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        torch.set_default_tensor_type(torch.DoubleTensor)
        result = torch.tensor([1.2, 3.8])
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
        torch.set_default_tensor_type(t="torch.FloatTensor")
        result = torch.tensor([1.2, 3.8])
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        torch.set_default_tensor_type(torch.FloatTensor)
        result = torch.tensor([1.2, 3.8])
        """
    )
    obj.run(pytorch_code, ["result"])


def _test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        torch.set_default_tensor_type("torch.BFloat16Tensor")
        result = True
        """
    )
    obj.run(pytorch_code, ["result"])


def _test_case_9():
    pytorch_code = textwrap.dedent(
        """
        import torch
        torch.set_default_tensor_type(torch.BFloat16Tensor)
        result = True
        """
    )
    obj.run(pytorch_code, ["result"])


def _test_case_10():
    pytorch_code = textwrap.dedent(
        """
        import torch
        torch.set_default_tensor_type("torch.cuda.DoubleTensor")
        result = torch.tensor([1.2, 3.8])
        """
    )
    obj.run(pytorch_code, ["result"])


def _test_case_11():
    pytorch_code = textwrap.dedent(
        """
        import torch
        torch.set_default_tensor_type(torch.cuda.DoubleTensor)
        result = torch.tensor([1.2, 3.8])
        """
    )
    obj.run(pytorch_code, ["result"])


def _test_case_12():
    pytorch_code = textwrap.dedent(
        """
        import torch
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        result = torch.tensor([1.2, 3.8])
        """
    )
    obj.run(pytorch_code, ["result"])


def _test_case_13():
    pytorch_code = textwrap.dedent(
        """
        import torch
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        result = torch.tensor([1.2, 3.8])
        """
    )
    obj.run(pytorch_code, ["result"])


def _test_case_14():
    pytorch_code = textwrap.dedent(
        """
        import torch
        torch.set_default_tensor_type("torch.cuda.HalfTensor")
        result = torch.tensor([1.2, 3.8])
        """
    )
    obj.run(pytorch_code, ["result"])


def _test_case_15():
    pytorch_code = textwrap.dedent(
        """
        import torch
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        result = torch.tensor([1.2, 3.8])
        """
    )
    obj.run(pytorch_code, ["result"])


def _test_case_16():
    pytorch_code = textwrap.dedent(
        """
        import torch
        torch.set_default_tensor_type("torch.cuda.BFloat16Tensor")
        result = True
        """
    )
    obj.run(pytorch_code, ["result"])


def _test_case_17():
    pytorch_code = textwrap.dedent(
        """
        import torch
        torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
        result = True
        """
    )
    obj.run(pytorch_code, ["result"])
