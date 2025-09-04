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

obj = APIBase("torch.Tensor.scatter")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.arange(15).reshape([3, 5]).type(torch.float32)
        index = torch.tensor([[0], [1], [2]])
        result = x.scatter(1, index, 1.0)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.arange(15).reshape([3, 5]).type(torch.float32)
        index = torch.tensor([[0], [1], [2]])
        result = x.scatter(dim=1, index=index, value=1.0)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.arange(15).reshape([3, 5]).type(torch.float32)
        index = torch.tensor([[0], [1], [2]])
        result = x.scatter(1, index, 1.0, reduce='multiply')

        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.arange(15).reshape([3, 5]).type(torch.float32)
        index = torch.tensor([[0], [1], [2]])
        result = x.scatter(dim=1, index=index, value=1.0, reduce='multiply')

        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.arange(15).reshape([3, 5]).type(torch.float32)
        index = torch.tensor([[0], [1], [2]])
        result = x.scatter(1, index, 1.0, reduce='add')

        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.arange(15).reshape([3, 5]).type(torch.float32)
        index = torch.tensor([[0], [1], [2]])
        result = x.scatter(dim=1, index=index, value=1.0, reduce='add')

        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import numpy as np
        np.random.seed(10)
        src_np = np.random.randn(3, 5).astype('float32')
        x = torch.arange(15).reshape([3, 5]).type(torch.float32)
        index = torch.tensor([[0], [1], [2]])
        src = torch.tensor(src_np)
        result = x.scatter(1, index, src)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import numpy as np
        np.random.seed(10)
        src_np = np.random.randn(3, 5).astype('float32')
        x = torch.arange(15).reshape([3, 5]).type(torch.float32)
        index = torch.tensor([[0], [1], [2]])
        src = torch.tensor(src_np)
        result = x.scatter(dim=1, index=index, src=src)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import numpy as np
        np.random.seed(10)
        src_np = np.random.randn(3, 5).astype('float32')
        x = torch.arange(15).reshape([3, 5]).type(torch.float32)
        index = torch.tensor([[0], [1], [2]])
        src = torch.tensor(src_np)
        result = x.scatter(1, index, src, reduce='add')
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_10():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import numpy as np
        np.random.seed(10)
        src_np = np.random.randn(3, 5).astype('float32')
        x = torch.arange(15).reshape([3, 5]).type(torch.float32)
        index = torch.tensor([[0], [1], [2]])
        src = torch.tensor(src_np)
        result = x.scatter(dim=1, index=index, src=src, reduce='add')
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_11():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import numpy as np
        np.random.seed(10)
        src_np = np.random.randn(3, 5).astype('float32')
        x = torch.arange(15).reshape([3, 5]).type(torch.float32)
        index = torch.tensor([[0], [1], [2]])
        src = torch.tensor(src_np)
        result = x.scatter(1, index, src, reduce='multiply')
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_12():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import numpy as np
        np.random.seed(10)
        src_np = np.random.randn(3, 5).astype('float32')
        x = torch.arange(15).reshape([3, 5]).type(torch.float32)
        index = torch.tensor([[0], [1], [2]])
        src = torch.tensor(src_np)
        result = x.scatter(dim=1, index=index, src=src, reduce='multiply')
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_13():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.arange(15).reshape([3, 5]).type(torch.float32)
        index = torch.tensor([[0, 1, 2], [3, 0, 1], [1, 2, 4]])
        result = x.scatter(index=index, src=torch.full([3, 3], -1.), dim=1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_14():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.arange(15).reshape([3, 5]).type(torch.float32)
        index = torch.tensor([[0, 1, 2], [3, 0, 1], [1, 2, 4]])
        result = x.scatter(dim=1, src=torch.full([3, 3], -1.), reduce='add', index=index)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_15():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.arange(15).reshape([3, 5]).type(torch.float32)
        index = torch.tensor([[0, 1, 2], [3, 0, 1], [1, 2, 4]])
        result = x.scatter(reduce='multiply', dim=1, index=index, src=torch.full([3, 3], -1.))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_16():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.arange(15).reshape([3, 5]).type(torch.float32)
        index = torch.tensor([[0, 1, 2], [3, 0, 1], [1, 2, 4]])
        result = x.scatter(index=index, value=10.0, dim=1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_17():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.arange(15).reshape([3, 5]).type(torch.float32)
        index = torch.tensor([[0, 1, 2], [3, 0, 1], [1, 2, 4]])
        result = x.scatter(dim=1, value=3.0, reduce='add', index=index)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_18():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.arange(15).reshape([3, 5]).type(torch.float32)
        index = torch.tensor([[0, 1, 2], [3, 0, 1], [1, 2, 4]])
        result = x.scatter(reduce='multiply', dim=1, index=index, value=-3.0)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_19_complex():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.arange(144).reshape([3, 4, 3, 4]).type(torch.float16)
        input.requires_grad = True
        index = torch.tensor([[[[1], [0]]], [[[0], [1]]]], dtype=torch.int64)
        src = torch.zeros_like(index).to(input.dtype)
        src.requires_grad = True
        result = input.scatter(0, index=index, src=src)
        result.backward(torch.ones_like(result))
        input_grad = input.grad
        src_grad = src.grad
        input_grad.requires_grad = False
        src_grad.requires_grad = False
        """
    )
    obj.run(pytorch_code, ["result", "input_grad", "src_grad"])


def test_case_20_complex():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.arange(225).reshape([3, 3, 5, 5]).type(torch.float32)
        input.requires_grad = True
        index = torch.arange(5).reshape(5, 1).repeat(2, 2, 1, 5)
        src = torch.ones_like(index).to(input.dtype)
        src.requires_grad = True
        result = input.scatter(2, index, src)
        result.backward(torch.ones_like(result))
        input_grad = input.grad
        src_grad = src.grad
        input_grad.requires_grad = False
        src_grad.requires_grad = False
        """
    )
    obj.run(pytorch_code, ["result", "input_grad", "src_grad"])


def test_case_21_complex():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.arange(240).reshape([2, 3, 2, 4, 5]).type(torch.int32)
        index = torch.ones([2, 1, 2, 3, 4], dtype = torch.int32)
        src = torch.full([2, 10, 3, 10, 6], -99, dtype = torch.int32)
        result = input.scatter(3, index, src=src, reduce = "add")
        """
    )
    obj.run(pytorch_code, ["result"])
