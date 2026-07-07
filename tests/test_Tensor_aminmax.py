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

obj = APIBase("torch.Tensor.aminmax")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
        result = t.aminmax()
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
        result = t.aminmax(dim=-1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
        result = t.aminmax(dim=-1, keepdim=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
        result = t.aminmax()[0]
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
        result = t.aminmax(keepdim=True, dim=-1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    # integer tensor support
    pytorch_code = textwrap.dedent(
        """
        import torch
        t = torch.tensor([[-12, 23, 34], [-45, 56, -67], [78, -89, 100]], dtype=torch.int64)
        result = t.aminmax()
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    # three-dimensional tensor along middle axis
    pytorch_code = textwrap.dedent(
        """
        import torch
        t = torch.tensor([
            [[1., 2., 3.], [9., 8., 7.]],
            [[4., 5., 6.], [10., 11., 12.]]], dtype=torch.float32)
        result = t.aminmax(dim=1, keepdim=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    # mix of positional and keyword args
    pytorch_code = textwrap.dedent(
        """
        import torch
        t = torch.tensor([[1.5, 2.5, 3.5]], dtype=torch.float64)
        result = t.aminmax(dim=-1, keepdim=False)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    # expression argument passed as keyword args
    pytorch_code = textwrap.dedent(
        """
        import torch
        t = torch.tensor([-3.0, -1.0, 0.0, 2.0])
        d = 1 - 1
        k = False or True
        result = t.aminmax(dim=d, keepdim=k)
        """
    )
    obj.run(pytorch_code, ["result"])
