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

obj = APIBase("torch.Tensor.addcmul")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        tensor1 = torch.tensor([1., 2., 3.])
        tensor2 = torch.tensor([4., 5., 6.])
        input = torch.tensor([7., 8., 9.])
        result = input.addcmul(tensor1, tensor2)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        tensor1 = torch.tensor([1., 2., 3.])
        tensor2 = torch.tensor([4., 5., 6.])
        input = torch.tensor([7., 8., 9.])
        result = input.addcmul(tensor1, tensor2, value=2)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        tensor1 = torch.tensor([1., 2., 3.])
        tensor2 = torch.tensor([4., 5., 6.])
        input = torch.tensor([7., 8., 9.])
        value = 5.0
        result = input.addcmul(tensor1, tensor2=tensor2, value=value)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        tensor1 = torch.tensor([1., 2., 3.])
        tensor2 = torch.tensor([4., 5., 6.])
        input = torch.tensor([7., 8., 9.])
        value = 5
        result = input.addcmul(tensor1, tensor2, value=value)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        tensor1 = torch.tensor([1., 2., 3.])
        tensor2 = torch.tensor([4., 5., 6.])
        input = torch.tensor([7., 8., 9.])
        value = 5
        result = input.addcmul(tensor1=tensor1, tensor2=tensor2, value=value)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        tensor1 = torch.tensor([1., 2., 3.])
        tensor2 = torch.tensor([4., 5., 6.])
        input = torch.tensor([7., 8., 9.])
        value = 5
        result = input.addcmul(tensor1=tensor1, value=value, tensor2=tensor2)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    """2D tensor test"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        tensor1 = torch.tensor([[1., 2.], [3., 4.]])
        tensor2 = torch.tensor([[5., 6.], [7., 8.]])
        input = torch.tensor([[1., 1.], [1., 1.]])
        result = input.addcmul(tensor1, tensor2, value=2)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    """3D tensor test"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        tensor1 = torch.tensor([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]])
        tensor2 = torch.tensor([[[1., 1.], [1., 1.]], [[1., 1.], [1., 1.]]])
        input = torch.tensor([[[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]]])
        result = input.addcmul(tensor1, tensor2)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    """float64 dtype test"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        tensor1 = torch.tensor([1., 2., 3.], dtype=torch.float64)
        tensor2 = torch.tensor([4., 5., 6.], dtype=torch.float64)
        input = torch.tensor([7., 8., 9.], dtype=torch.float64)
        result = input.addcmul(tensor1, tensor2, value=2.0)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_10():
    """gradient computation test"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        tensor1 = torch.tensor([1., 2., 3.], requires_grad=True)
        tensor2 = torch.tensor([4., 5., 6.], requires_grad=True)
        input = torch.tensor([7., 8., 9.], requires_grad=True)
        result = input.addcmul(tensor1, tensor2, value=2.0)
        result.sum().backward()
        input_grad = input.grad
        """
    )
    obj.run(pytorch_code, ["result", "input_grad"], check_stop_gradient=False)


def test_case_11():
    """expression argument test"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        tensor1 = torch.tensor([1., 2., 3.])
        tensor2 = torch.tensor([4., 5., 6.])
        input = torch.tensor([7., 8., 9.])
        result = input.addcmul(tensor1, tensor2, value=1 + 1)
        """
    )
    obj.run(pytorch_code, ["result"])
