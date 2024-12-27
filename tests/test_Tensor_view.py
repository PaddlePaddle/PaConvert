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

obj = APIBase("torch.Tensor.view")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.arange(4.)
        result = a.view(2, 2)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.arange(4.)
        result = a.view((2, 2))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.arange(9.)
        result = a.view([3, 3])
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.arange(9)
        shape = (3, 3)
        result = a.view(shape)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.arange(9)
        shape = (3, 3)
        result = a.view(*shape)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.arange(4.)
        k = 2
        result = a.view((k, k))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.arange(2.)
        k = 2
        result = a.view(k)
        """
    )
    obj.run(pytorch_code, ["result"])


# # Because the current paddle.view does not support tensors without a shape, this case cannot run properly.
# def test_case_9():
#     pytorch_code = textwrap.dedent(
#         """
#         import torch
#         a = torch.tensor(1.)
#         result = a.view(1)
#         """
#     )
#     obj.run(pytorch_code, ["result"])


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.arange(6.)
        result = a.view((2, 3))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.arange(6.)
        result = a.view(torch.int32)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_10():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.arange(6.)
        k = torch.int32
        result = a.view(k)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_11():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.arange(6.)
        k = torch.int32
        result = a.view(dtype = k)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_12():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.arange(6)
        result = a.view(dtype = torch.int32)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_13():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.arange(6)
        result = a.view(dtype = torch.float32)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_14():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.arange(6)
        result = a.view(dtype = torch.cfloat)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_15():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.arange(6)
        result = a.view(dtype = torch.half)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_16():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.arange(6)
        result = a.view(dtype = torch.bool)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_17():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.arange(6)
        result = a.view(size = [2,3])
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_18():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.arange(6)
        k = (2,3)
        result = a.view(size = k)
        """
    )
    obj.run(pytorch_code, ["result"])


# add Infermeta test case
def test_case_19():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.arange(2*3*4*5)
        result = a.view(-1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_20():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.arange(2*3*4*5)
        result = a.view(2,-1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_21():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.arange(2*3*4*5)
        result = a.view(2,3,-1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_22():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.arange(2*3*4*5)
        result = a.view(2,3,4,-1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_23():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.arange(2*3*4*5)
        result = a.view(2*3,-1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_24():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.arange(2*3*4*5)
        result = a.view(3*4,-1)
        """
    )
    obj.run(pytorch_code, ["result"])
