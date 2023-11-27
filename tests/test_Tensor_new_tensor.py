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

obj = APIBase("torch.Tensor.new_tensor")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1., 2., 3.])
        data = [[0, 1], [2, 3]]
        result = x.new_tensor(data)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1., 2., 3.])
        data = [[0, 1], [2, 3]]
        result = x.new_tensor(data, dtype=torch.float64)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1., 2., 3.])
        data = [[0, 1], [2, 3]]
        result = x.new_tensor(data, device='cpu')
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1., 2., 3.])
        data = [[0, 1], [2, 3]]
        result = x.new_tensor(data, device='cpu', requires_grad=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1., 2., 3.])
        data = [[0, 1], [2, 3]]
        result = x.new_tensor(data, device='cpu', requires_grad=False)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1., 2., 3.])
        data = [[0, 1], [2, 3]]
        result = x.new_tensor(data, dtype=torch.float64, device='cpu', requires_grad=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1., 2., 3.])
        data = [[0, 1], [2, 3]]
        result = x.new_tensor(data=data, dtype=torch.float64, device='cpu', requires_grad=True)
        """
    )
    obj.run(pytorch_code, ["result"])


# torch.Tensor.new_tensor not support parameters `layout` and `pin_memory`
# these two keyword parameters value is set to default.
def _test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1., 2., 3.])
        data = [[0, 1], [2, 3]]
        result = x.new_tensor(data=data, dtype=torch.float64, device='cpu', requires_grad=True, layout=torch.strided, pin_memory=False)
        """
    )
    obj.run(pytorch_code, ["result"])
