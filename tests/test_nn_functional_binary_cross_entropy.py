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

obj = APIBase("torch.nn.functional.binary_cross_entropy")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input = torch.tensor([[0.2837, 0.0297,  0.0355],
            [ 0.9112, 0.7526, 0.4061]])
        target = torch.tensor([[1.,0.,1.],[0.,1.,0.]])
        weight = torch.tensor([0.5,0.2,0.3])
        result = torch.nn.functional.binary_cross_entropy(input,target,weight=weight,size_average=True)
        """
    )
    expect_paddle_code = textwrap.dedent(
        """
        import paddle

        input = paddle.tensor([[0.2837, 0.0297, 0.0355], [0.9112, 0.7526, 0.4061]])
        target = paddle.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        weight = paddle.tensor([0.5, 0.2, 0.3])
        result = paddle.nn.functional.binary_cross_entropy(
            input, target, weight=weight, size_average=True
        )
        """
    )
    obj.run(pytorch_code, expect_paddle_code=expect_paddle_code)


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input = torch.tensor([[0.2837, 0.0297,  0.0355],
            [ 0.9112, 0.7526, 0.4061]])
        target = torch.tensor([[1.,0.,1.],[0.,1.,0.]])
        weight = torch.tensor([0.5,0.2,0.3])
        result = torch.nn.functional.binary_cross_entropy(input,target,weight=weight,size_average=False)
        """
    )
    expect_paddle_code = textwrap.dedent(
        """
        import paddle

        input = paddle.tensor([[0.2837, 0.0297, 0.0355], [0.9112, 0.7526, 0.4061]])
        target = paddle.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        weight = paddle.tensor([0.5, 0.2, 0.3])
        result = paddle.nn.functional.binary_cross_entropy(
            input, target, weight=weight, size_average=False
        )
        """
    )
    obj.run(pytorch_code, expect_paddle_code=expect_paddle_code)


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input = torch.tensor([[0.2837, 0.0297,  0.0355],
            [ 0.9112, 0.7526, 0.4061]])
        target = torch.tensor([[1.,0.,1.],[0.,1.,0.]])
        weight = torch.tensor([0.5,0.2,0.3])
        result = torch.nn.functional.binary_cross_entropy(input,target,weight=weight,reduction='none')
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input = torch.tensor([[0.2837, 0.0297,  0.0355],
            [ 0.9112, 0.7526, 0.4061]])
        target = torch.tensor([[1.,0.,1.],[0.,1.,0.]])
        weight = torch.tensor([0.5,0.2,0.3])
        result = torch.nn.functional.binary_cross_entropy(input,target,weight=weight,reduction='mean')
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input = torch.tensor([[0.2837, 0.0297,  0.0355],
            [ 0.9112, 0.7526, 0.4061]])
        target = torch.tensor([[1.,0.,1.],[0.,1.,0.]])
        weight = torch.tensor([0.5,0.2,0.3])
        result = torch.nn.functional.binary_cross_entropy(input,target,weight=weight,reduction='sum')
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input = torch.tensor([[0.2837, 0.0297,  0.0355],
            [ 0.9112, 0.7526, 0.4061]])
        target = torch.tensor([[1.,0.,1.],[0.,1.,0.]])
        weight = torch.tensor([0.5,0.2,0.3])
        result = torch.nn.functional.binary_cross_entropy(input,target,weight=weight,reduce=True)
        """
    )
    expect_paddle_code = textwrap.dedent(
        """
        import paddle

        input = paddle.tensor([[0.2837, 0.0297, 0.0355], [0.9112, 0.7526, 0.4061]])
        target = paddle.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        weight = paddle.tensor([0.5, 0.2, 0.3])
        result = paddle.nn.functional.binary_cross_entropy(
            input, target, weight=weight, reduce=True
        )
        """
    )
    obj.run(pytorch_code, expect_paddle_code=expect_paddle_code)


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input = torch.tensor([[0.2837, 0.0297,  0.0355],
            [ 0.9112, 0.7526, 0.4061]])
        target = torch.tensor([[1.,0.,1.],[0.,1.,0.]])
        weight = torch.tensor([0.5,0.2,0.3])
        result = torch.nn.functional.binary_cross_entropy(input,target,weight=weight,reduce=False)
        """
    )
    expect_paddle_code = textwrap.dedent(
        """
        import paddle

        input = paddle.tensor([[0.2837, 0.0297, 0.0355], [0.9112, 0.7526, 0.4061]])
        target = paddle.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        weight = paddle.tensor([0.5, 0.2, 0.3])
        result = paddle.nn.functional.binary_cross_entropy(
            input, target, weight=weight, reduce=False
        )
        """
    )
    obj.run(pytorch_code, expect_paddle_code=expect_paddle_code)


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input = torch.tensor([[0.2837, 0.0297,  0.0355],
            [ 0.9112, 0.7526, 0.4061]])
        target = torch.tensor([[1.,0.,1.],[0.,1.,0.]])
        result = torch.nn.functional.binary_cross_entropy(input,target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input = torch.tensor([[0.2837, 0.0297,  0.0355],
            [ 0.9112, 0.7526, 0.4061]])
        target = torch.tensor([[1.,0.,1.],[0.,1.,0.]])
        weight = torch.tensor([0.5,0.2,0.3])
        result = torch.nn.functional.binary_cross_entropy(input,target,weight=weight, size_average=None, reduce=False, reduction='mean')
        """
    )
    expect_paddle_code = textwrap.dedent(
        """
        import paddle

        input = paddle.tensor([[0.2837, 0.0297, 0.0355], [0.9112, 0.7526, 0.4061]])
        target = paddle.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        weight = paddle.tensor([0.5, 0.2, 0.3])
        result = paddle.nn.functional.binary_cross_entropy(
            input, target, weight=weight, size_average=None, reduce=False, reduction="mean"
        )
        """
    )
    obj.run(pytorch_code, expect_paddle_code=expect_paddle_code)


def test_case_10():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input = torch.tensor([[0.2837, 0.0297,  0.0355],
            [ 0.9112, 0.7526, 0.4061]])
        target = torch.tensor([[1.,0.,1.],[0.,1.,0.]])
        result = torch.nn.functional.binary_cross_entropy(input=input, target=target)
        """
    )
    expect_paddle_code = textwrap.dedent(
        """
        import paddle

        input = paddle.tensor([[0.2837, 0.0297, 0.0355], [0.9112, 0.7526, 0.4061]])
        target = paddle.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        result = paddle.nn.functional.binary_cross_entropy(input=input, target=target)
        """
    )
    obj.run(pytorch_code, expect_paddle_code=expect_paddle_code)


def test_case_11():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input = torch.tensor([[0.2837, 0.0297,  0.0355],
            [ 0.9112, 0.7526, 0.4061]])
        target = torch.tensor([[1.,0.,1.],[0.,1.,0.]])
        weight = torch.tensor([0.5,0.2,0.3])
        result = torch.nn.functional.binary_cross_entropy(target=target, input=input, reduction='sum', weight=weight)
        """
    )
    expect_paddle_code = textwrap.dedent(
        """
        import paddle

        input = paddle.tensor([[0.2837, 0.0297, 0.0355], [0.9112, 0.7526, 0.4061]])
        target = paddle.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        weight = paddle.tensor([0.5, 0.2, 0.3])
        result = paddle.nn.functional.binary_cross_entropy(
            target=target, input=input, reduction="sum", weight=weight
        )
        """
    )
    obj.run(pytorch_code, expect_paddle_code=expect_paddle_code)


def test_case_12():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input = torch.tensor([[0.2837, 0.0297,  0.0355],
            [ 0.9112, 0.7526, 0.4061]])
        target = torch.tensor([[1.,0.,1.],[0.,1.,0.]])
        weight = torch.tensor([0.5,0.2,0.3])
        result = torch.nn.functional.binary_cross_entropy(input=input, target=target, weight=weight, size_average=None, reduce=False, reduction='mean')
        """
    )
    expect_paddle_code = textwrap.dedent(
        """
        import paddle

        input = paddle.tensor([[0.2837, 0.0297, 0.0355], [0.9112, 0.7526, 0.4061]])
        target = paddle.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        weight = paddle.tensor([0.5, 0.2, 0.3])
        result = paddle.nn.functional.binary_cross_entropy(
            input=input,
            target=target,
            weight=weight,
            size_average=None,
            reduce=False,
            reduction="mean",
        )
        """
    )
    obj.run(pytorch_code, expect_paddle_code=expect_paddle_code)


def test_case_13():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        input = torch.tensor([[0.2837, 0.0297,  0.0355],
            [ 0.9112, 0.7526, 0.4061]])
        target = torch.tensor([[1.,0.,1.],[0.,1.,0.]])
        weight = torch.tensor([0.5,0.2,0.3])
        result = torch.nn.functional.binary_cross_entropy(reduction='mean', reduce=False, size_average=None, weight=weight, target=target, input=input)
        """
    )
    expect_paddle_code = textwrap.dedent(
        """
        import paddle

        input = paddle.tensor([[0.2837, 0.0297, 0.0355], [0.9112, 0.7526, 0.4061]])
        target = paddle.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        weight = paddle.tensor([0.5, 0.2, 0.3])
        result = paddle.nn.functional.binary_cross_entropy(
            reduction="mean",
            reduce=False,
            size_average=None,
            weight=weight,
            target=target,
            input=input,
        )
        """
    )
    obj.run(pytorch_code, expect_paddle_code=expect_paddle_code)
