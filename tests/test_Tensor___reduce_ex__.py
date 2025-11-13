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

obj = APIBase("torch.Tensor.__reduce_ex__")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import pickle
        x = torch.tensor([1, 2, 3], device=torch.device('cpu'), dtype=torch.int64, requires_grad=False)
        data = pickle.dumps(x)
        result = pickle.loads(data)
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
        import pickle
        x = torch.tensor([1, 2, 3], device=torch.device('cuda'), dtype=torch.int64, pin_memory=False, requires_grad=False)
        data = pickle.dumps(x)
        result = pickle.loads(data)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import pickle
        x = torch.tensor([4., 5., 6.], device=torch.device('cpu'), dtype=torch.float32, requires_grad=True)
        data = pickle.dumps(x)
        result = pickle.loads(data)
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
        import pickle
        x = torch.tensor([4., 5., 6.], device=torch.device('cuda:0'), dtype=torch.float32, pin_memory=True, requires_grad=True)
        data = pickle.dumps(x)
        result = pickle.loads(data)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import pickle
        x = torch.tensor([4., 5., 6.], dtype=torch.float64, requires_grad=True)
        data = pickle.dumps(x)
        result = pickle.loads(data)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import pickle
        x = torch.tensor([4., 5., 6.], dtype=torch.float64, requires_grad=True)
        data = pickle.dumps(x)
        result = pickle.loads(data)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import pickle
        x = torch.tensor([True, False, True])
        data = pickle.dumps(x)
        result = pickle.loads(data)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import pickle
        x = torch.tensor([7., 8., 9.])
        data = pickle.dumps(x)
        result = pickle.loads(data)
        """
    )
    obj.run(pytorch_code, ["result"])
