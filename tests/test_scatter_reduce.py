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

obj = APIBase("torch.scatter_reduce")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        src = torch.tensor([1., 2., 3., 4., 5., 6.])
        index = torch.tensor([0, 1, 0, 1, 2, 1])
        input = torch.tensor([1., 2., 3., 4.])
        type = "sum"
        result = torch.scatter_reduce(input, 0, index, src, reduce=type)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        src = torch.tensor([1., 2., 3., 4., 5., 6.])
        index = torch.tensor([0, 1, 0, 1, 2, 1])
        input = torch.tensor([1., 2., 3., 4.])
        result = torch.scatter_reduce(input=input, dim=0, index=index, src=src, reduce="sum", include_self=False)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        src = torch.tensor([1., 2., 3., 4., 5., 6.])
        index = torch.tensor([0, 1, 0, 1, 2, 1])
        input = torch.tensor([1., 2., 3., 4.])
        result = torch.scatter_reduce(input, 0, index, src, "amax")
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        src = torch.tensor([1., 2., 3., 4., 5., 6.])
        index = torch.tensor([0, 1, 0, 1, 2, 1])
        input = torch.tensor([1., 2., 3., 4.])
        result = torch.scatter_reduce(input, 0, index, src, reduce="amin")
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        src = torch.tensor([1., 2., 3., 4., 5., 6.])
        index = torch.tensor([0, 1, 0, 1, 2, 1])
        input = torch.tensor([1., 2., 3., 4.])
        re_type = "prod"
        result = torch.scatter_reduce(input, 0, index, src, reduce=re_type)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        src = torch.tensor([[1., 2.],[3., 4.]])
        index = torch.tensor([[0, 0], [0, 0]])
        input = torch.tensor([[10., 30., 20.], [60., 40., 50.]])
        result = torch.scatter_reduce(input, 0, index, src, reduce="sum")
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        src = torch.tensor([[1., 2.],[3., 4.]])
        index = torch.tensor([[0, 0], [0, 0]])
        input = torch.tensor([[10., 30., 20.], [60., 40., 50.]])
        result = torch.scatter_reduce(input, 0, index, src, reduce="prod", include_self=False)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        src = torch.tensor([[1., 2.],[3., 4.]])
        index = torch.tensor([[0, 0], [0, 0]])
        input = torch.tensor([[10., 30., 20.], [60., 40., 50.]])
        re_type = "prod"
        result = torch.scatter_reduce(input, 0, index, src, re_type)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    pytorch_code = textwrap.dedent(
        """
        import torch
        src = torch.tensor([[1., 2., 3.],[4., 5., 6.]])
        index = torch.tensor([[0, 1, 2], [0, 2, 1]])
        input = torch.tensor([[10., 30., 20.], [60., 40., 50.]])
        re_type = "prod"
        result = torch.scatter_reduce(input, 1, index, src, re_type, include_self=False)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_10():
    pytorch_code = textwrap.dedent(
        """
        import torch
        src = torch.tensor([[1., 2., 3.],[4., 5., 6.]])
        index = torch.tensor([[0, 1, 2], [0, 2, 1]])
        input = torch.tensor([[10., 30., 20.], [60., 40., 50.]])
        re_type = "sum"
        result = torch.scatter_reduce(input=input, dim=1, index=index, src=src, reduce=re_type, include_self=False)
        """
    )
    obj.run(pytorch_code, ["result"])


# current paddle.scatter_reduce has some bug due to the implementation of put_along_axis api
def test_case_11():
    pytorch_code = textwrap.dedent(
        """
        import torch
        src = torch.tensor([[1., 2., 3.],[4., 5., 6.]])
        index = torch.tensor([[0, 1, 2], [0, 2, 1]])
        input = torch.tensor([[10., 30., 20.], [60., 40., 50.]])
        re_type = "mean"
        result = torch.scatter_reduce(input, 1, index, src, re_type)
        """
    )
    obj.run(pytorch_code, ["result"])
