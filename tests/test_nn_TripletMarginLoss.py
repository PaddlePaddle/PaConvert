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

obj = APIBase("torch.nn.TripletMarginLoss")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, 5, 3], [0, 3, 2], [1, 4, 1]]).type(torch.float32)
        positive= torch.Tensor([[5, 1, 2], [3, 2, 1], [3, -1, 1]]).type(torch.float32)
        negative = torch.Tensor([[2, 1, -3], [1, 1, -1], [4, -2, 1]]).type(torch.float32)
        cri = torch.nn.TripletMarginLoss()
        result = cri(input, positive, negative)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, 5, 3], [0, 3, 2], [1, 4, 1]]).type(torch.float32)
        positive= torch.Tensor([[5, 1, 2], [3, 2, 1], [3, -1, 1]]).type(torch.float32)
        negative = torch.Tensor([[2, 1, -3], [1, 1, -1], [4, -2, 1]]).type(torch.float32)
        cri = torch.nn.TripletMarginLoss(1.3, 3.2, 1e-5, False, True, False, reduction='mean')
        result = cri(input, positive, negative)
        """
    )
    obj.run(pytorch_code, ["result"], rtol=1.0e-5, atol=1.0e-8)


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, 5, 3], [0, 3, 2], [1, 4, 1]]).type(torch.float32)
        positive= torch.Tensor([[5, 1, 2], [3, 2, 1], [3, -1, 1]]).type(torch.float32)
        negative = torch.Tensor([[2, 1, -3], [1, 1, -1], [4, -2, 1]]).type(torch.float32)
        cri = torch.nn.TripletMarginLoss(1.3, 3.2, 1e-5, True, True, True, reduction='sum')
        result = cri(input, positive, negative)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, 5, 3], [0, 3, 2], [1, 4, 1]]).type(torch.float32)
        positive= torch.Tensor([[5, 1, 2], [3, 2, 1], [3, -1, 1]]).type(torch.float32)
        negative = torch.Tensor([[2, 1, -3], [1, 1, -1], [4, -2, 1]]).type(torch.float32)
        cri = torch.nn.TripletMarginLoss(1.3, 3.2, 1e-5, False, False, True, reduction='mean')
        result = cri(input, positive, negative)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, 5, 3], [0, 3, 2], [1, 4, 1]]).type(torch.float32)
        positive= torch.Tensor([[5, 1, 2], [3, 2, 1], [3, -1, 1]]).type(torch.float32)
        negative = torch.Tensor([[2, 1, -3], [1, 1, -1], [4, -2, 1]]).type(torch.float32)
        cri = torch.nn.TripletMarginLoss(1.3, 3.2, 1e-5, True, True, False, reduction='none')
        result = cri(input, positive, negative)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, 5, 3], [0, 3, 2], [1, 4, 1]]).type(torch.float32)
        positive= torch.Tensor([[5, 1, 2], [3, 2, 1], [3, -1, 1]]).type(torch.float32)
        negative = torch.Tensor([[2, 1, -3], [1, 1, -1], [4, -2, 1]]).type(torch.float32)
        cri = torch.nn.TripletMarginLoss(1.3, 3.2, 1e-5, False, False, False, reduction='sum')
        result = cri(input, positive, negative)
        """
    )
    obj.run(pytorch_code, ["result"], rtol=1.0e-5, atol=1.0e-8)


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, 5, 3], [0, 3, 2], [1, 4, 1]]).type(torch.float32)
        positive= torch.Tensor([[5, 1, 2], [3, 2, 1], [3, -1, 1]]).type(torch.float32)
        negative = torch.Tensor([[2, 1, -3], [1, 1, -1], [4, -2, 1]]).type(torch.float32)
        cri = torch.nn.TripletMarginLoss(1.3, 3.2, 1e-5, True, True, False, reduction='mean')
        result = cri(input, positive, negative)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, 5, 3], [0, 3, 2], [1, 4, 1]]).type(torch.float32)
        positive= torch.Tensor([[5, 1, 2], [3, 2, 1], [3, -1, 1]]).type(torch.float32)
        negative = torch.Tensor([[2, 1, -3], [1, 1, -1], [4, -2, 1]]).type(torch.float32)
        cri = torch.nn.TripletMarginLoss(1.3, 3.2, 1e-5, True, False, True, reduction='mean')
        result = cri(input, positive, negative)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, 5, 3], [0, 3, 2], [1, 4, 1]]).type(torch.float32)
        positive= torch.Tensor([[5, 1, 2], [3, 2, 1], [3, -1, 1]]).type(torch.float32)
        negative = torch.Tensor([[2, 1, -3], [1, 1, -1], [4, -2, 1]]).type(torch.float32)
        cri = torch.nn.TripletMarginLoss(1.3, 3.2, 1e-1, False, True, False, reduction='mean')
        result = cri(input, positive, negative)
        """
    )
    obj.run(pytorch_code, ["result"], rtol=1.0e-5, atol=1.0e-8)


# generated by validate_unittest autofix, based on test_case_2
def test_case_10():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, 5, 3], [0, 3, 2], [1, 4, 1]]).type(torch.float32)
        positive= torch.Tensor([[5, 1, 2], [3, 2, 1], [3, -1, 1]]).type(torch.float32)
        negative = torch.Tensor([[2, 1, -3], [1, 1, -1], [4, -2, 1]]).type(torch.float32)
        cri = torch.nn.TripletMarginLoss(1.3, 3.2, 1e-5, False, True, False, 'mean')
        result = cri(input, positive, negative)
        """
    )
    obj.run(pytorch_code, ["result"], rtol=1.0e-5, atol=1.0e-8)


# generated by validate_unittest autofix, based on test_case_2
def test_case_11():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, 5, 3], [0, 3, 2], [1, 4, 1]]).type(torch.float32)
        positive= torch.Tensor([[5, 1, 2], [3, 2, 1], [3, -1, 1]]).type(torch.float32)
        negative = torch.Tensor([[2, 1, -3], [1, 1, -1], [4, -2, 1]]).type(torch.float32)
        cri = torch.nn.TripletMarginLoss(margin=1.3, p=3.2, eps=1e-5, swap=False, size_average=True, reduce=False, reduction='mean')
        result = cri(input, positive, negative)
        """
    )
    obj.run(pytorch_code, ["result"], rtol=1.0e-5, atol=1.0e-8)


# generated by validate_unittest autofix, based on test_case_2
def test_case_12():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, 5, 3], [0, 3, 2], [1, 4, 1]]).type(torch.float32)
        positive= torch.Tensor([[5, 1, 2], [3, 2, 1], [3, -1, 1]]).type(torch.float32)
        negative = torch.Tensor([[2, 1, -3], [1, 1, -1], [4, -2, 1]]).type(torch.float32)
        cri = torch.nn.TripletMarginLoss(reduction='mean', reduce=False, size_average=True, swap=False, eps=1e-5, p=3.2, margin=1.3)
        result = cri(input, positive, negative)
        """
    )
    obj.run(pytorch_code, ["result"], rtol=1.0e-5, atol=1.0e-8)


# generated by validate_unittest autofix, based on test_case_3
def test_case_13():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, 5, 3], [0, 3, 2], [1, 4, 1]]).type(torch.float32)
        positive= torch.Tensor([[5, 1, 2], [3, 2, 1], [3, -1, 1]]).type(torch.float32)
        negative = torch.Tensor([[2, 1, -3], [1, 1, -1], [4, -2, 1]]).type(torch.float32)
        cri = torch.nn.TripletMarginLoss(1.3, 3.2, 1e-5, True, True, True, 'sum')
        result = cri(input, positive, negative)
        """
    )
    obj.run(pytorch_code, ["result"])


# generated by validate_unittest autofix, based on test_case_3
def test_case_14():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, 5, 3], [0, 3, 2], [1, 4, 1]]).type(torch.float32)
        positive= torch.Tensor([[5, 1, 2], [3, 2, 1], [3, -1, 1]]).type(torch.float32)
        negative = torch.Tensor([[2, 1, -3], [1, 1, -1], [4, -2, 1]]).type(torch.float32)
        cri = torch.nn.TripletMarginLoss(margin=1.3, p=3.2, eps=1e-5, swap=True, size_average=True, reduce=True, reduction='sum')
        result = cri(input, positive, negative)
        """
    )
    obj.run(pytorch_code, ["result"])


# generated by validate_unittest autofix, based on test_case_3
def test_case_15():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, 5, 3], [0, 3, 2], [1, 4, 1]]).type(torch.float32)
        positive= torch.Tensor([[5, 1, 2], [3, 2, 1], [3, -1, 1]]).type(torch.float32)
        negative = torch.Tensor([[2, 1, -3], [1, 1, -1], [4, -2, 1]]).type(torch.float32)
        cri = torch.nn.TripletMarginLoss(reduction='sum', reduce=True, size_average=True, swap=True, eps=1e-5, p=3.2, margin=1.3)
        result = cri(input, positive, negative)
        """
    )
    obj.run(pytorch_code, ["result"])


# generated by validate_unittest autofix, based on test_case_4
def test_case_16():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, 5, 3], [0, 3, 2], [1, 4, 1]]).type(torch.float32)
        positive= torch.Tensor([[5, 1, 2], [3, 2, 1], [3, -1, 1]]).type(torch.float32)
        negative = torch.Tensor([[2, 1, -3], [1, 1, -1], [4, -2, 1]]).type(torch.float32)
        cri = torch.nn.TripletMarginLoss(1.3, 3.2, 1e-5, False, False, True, 'mean')
        result = cri(input, positive, negative)
        """
    )
    obj.run(pytorch_code, ["result"])


# generated by validate_unittest autofix, based on test_case_4
def test_case_17():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, 5, 3], [0, 3, 2], [1, 4, 1]]).type(torch.float32)
        positive= torch.Tensor([[5, 1, 2], [3, 2, 1], [3, -1, 1]]).type(torch.float32)
        negative = torch.Tensor([[2, 1, -3], [1, 1, -1], [4, -2, 1]]).type(torch.float32)
        cri = torch.nn.TripletMarginLoss(margin=1.3, p=3.2, eps=1e-5, swap=False, size_average=False, reduce=True, reduction='mean')
        result = cri(input, positive, negative)
        """
    )
    obj.run(pytorch_code, ["result"])


# generated by validate_unittest autofix, based on test_case_4
def test_case_18():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, 5, 3], [0, 3, 2], [1, 4, 1]]).type(torch.float32)
        positive= torch.Tensor([[5, 1, 2], [3, 2, 1], [3, -1, 1]]).type(torch.float32)
        negative = torch.Tensor([[2, 1, -3], [1, 1, -1], [4, -2, 1]]).type(torch.float32)
        cri = torch.nn.TripletMarginLoss(reduction='mean', reduce=True, size_average=False, swap=False, eps=1e-5, p=3.2, margin=1.3)
        result = cri(input, positive, negative)
        """
    )
    obj.run(pytorch_code, ["result"])


# generated by validate_unittest autofix, based on test_case_5
def test_case_19():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, 5, 3], [0, 3, 2], [1, 4, 1]]).type(torch.float32)
        positive= torch.Tensor([[5, 1, 2], [3, 2, 1], [3, -1, 1]]).type(torch.float32)
        negative = torch.Tensor([[2, 1, -3], [1, 1, -1], [4, -2, 1]]).type(torch.float32)
        cri = torch.nn.TripletMarginLoss(1.3, 3.2, 1e-5, True, True, False, 'none')
        result = cri(input, positive, negative)
        """
    )
    obj.run(pytorch_code, ["result"])


# generated by validate_unittest autofix, based on test_case_5
def test_case_20():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, 5, 3], [0, 3, 2], [1, 4, 1]]).type(torch.float32)
        positive= torch.Tensor([[5, 1, 2], [3, 2, 1], [3, -1, 1]]).type(torch.float32)
        negative = torch.Tensor([[2, 1, -3], [1, 1, -1], [4, -2, 1]]).type(torch.float32)
        cri = torch.nn.TripletMarginLoss(margin=1.3, p=3.2, eps=1e-5, swap=True, size_average=True, reduce=False, reduction='none')
        result = cri(input, positive, negative)
        """
    )
    obj.run(pytorch_code, ["result"])


# generated by validate_unittest autofix, based on test_case_5
def test_case_21():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, 5, 3], [0, 3, 2], [1, 4, 1]]).type(torch.float32)
        positive= torch.Tensor([[5, 1, 2], [3, 2, 1], [3, -1, 1]]).type(torch.float32)
        negative = torch.Tensor([[2, 1, -3], [1, 1, -1], [4, -2, 1]]).type(torch.float32)
        cri = torch.nn.TripletMarginLoss(reduction='none', reduce=False, size_average=True, swap=True, eps=1e-5, p=3.2, margin=1.3)
        result = cri(input, positive, negative)
        """
    )
    obj.run(pytorch_code, ["result"])


# generated by validate_unittest autofix, based on test_case_6
def test_case_22():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, 5, 3], [0, 3, 2], [1, 4, 1]]).type(torch.float32)
        positive= torch.Tensor([[5, 1, 2], [3, 2, 1], [3, -1, 1]]).type(torch.float32)
        negative = torch.Tensor([[2, 1, -3], [1, 1, -1], [4, -2, 1]]).type(torch.float32)
        cri = torch.nn.TripletMarginLoss(1.3, 3.2, 1e-5, False, False, False, 'sum')
        result = cri(input, positive, negative)
        """
    )
    obj.run(pytorch_code, ["result"], rtol=1.0e-5, atol=1.0e-8)


# generated by validate_unittest autofix, based on test_case_6
def test_case_23():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, 5, 3], [0, 3, 2], [1, 4, 1]]).type(torch.float32)
        positive= torch.Tensor([[5, 1, 2], [3, 2, 1], [3, -1, 1]]).type(torch.float32)
        negative = torch.Tensor([[2, 1, -3], [1, 1, -1], [4, -2, 1]]).type(torch.float32)
        cri = torch.nn.TripletMarginLoss(margin=1.3, p=3.2, eps=1e-5, swap=False, size_average=False, reduce=False, reduction='sum')
        result = cri(input, positive, negative)
        """
    )
    obj.run(pytorch_code, ["result"], rtol=1.0e-5, atol=1.0e-8)


# generated by validate_unittest autofix, based on test_case_6
def test_case_24():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, 5, 3], [0, 3, 2], [1, 4, 1]]).type(torch.float32)
        positive= torch.Tensor([[5, 1, 2], [3, 2, 1], [3, -1, 1]]).type(torch.float32)
        negative = torch.Tensor([[2, 1, -3], [1, 1, -1], [4, -2, 1]]).type(torch.float32)
        cri = torch.nn.TripletMarginLoss(reduction='sum', reduce=False, size_average=False, swap=False, eps=1e-5, p=3.2, margin=1.3)
        result = cri(input, positive, negative)
        """
    )
    obj.run(pytorch_code, ["result"], rtol=1.0e-5, atol=1.0e-8)


# generated by validate_unittest autofix, based on test_case_7
def test_case_25():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, 5, 3], [0, 3, 2], [1, 4, 1]]).type(torch.float32)
        positive= torch.Tensor([[5, 1, 2], [3, 2, 1], [3, -1, 1]]).type(torch.float32)
        negative = torch.Tensor([[2, 1, -3], [1, 1, -1], [4, -2, 1]]).type(torch.float32)
        cri = torch.nn.TripletMarginLoss(1.3, 3.2, 1e-5, True, True, False, 'mean')
        result = cri(input, positive, negative)
        """
    )
    obj.run(pytorch_code, ["result"])


# generated by validate_unittest autofix, based on test_case_7
def test_case_26():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, 5, 3], [0, 3, 2], [1, 4, 1]]).type(torch.float32)
        positive= torch.Tensor([[5, 1, 2], [3, 2, 1], [3, -1, 1]]).type(torch.float32)
        negative = torch.Tensor([[2, 1, -3], [1, 1, -1], [4, -2, 1]]).type(torch.float32)
        cri = torch.nn.TripletMarginLoss(margin=1.3, p=3.2, eps=1e-5, swap=True, size_average=True, reduce=False, reduction='mean')
        result = cri(input, positive, negative)
        """
    )
    obj.run(pytorch_code, ["result"])


# generated by validate_unittest autofix, based on test_case_7
def test_case_27():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, 5, 3], [0, 3, 2], [1, 4, 1]]).type(torch.float32)
        positive= torch.Tensor([[5, 1, 2], [3, 2, 1], [3, -1, 1]]).type(torch.float32)
        negative = torch.Tensor([[2, 1, -3], [1, 1, -1], [4, -2, 1]]).type(torch.float32)
        cri = torch.nn.TripletMarginLoss(reduction='mean', reduce=False, size_average=True, swap=True, eps=1e-5, p=3.2, margin=1.3)
        result = cri(input, positive, negative)
        """
    )
    obj.run(pytorch_code, ["result"])


# generated by validate_unittest autofix, based on test_case_8
def test_case_28():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, 5, 3], [0, 3, 2], [1, 4, 1]]).type(torch.float32)
        positive= torch.Tensor([[5, 1, 2], [3, 2, 1], [3, -1, 1]]).type(torch.float32)
        negative = torch.Tensor([[2, 1, -3], [1, 1, -1], [4, -2, 1]]).type(torch.float32)
        cri = torch.nn.TripletMarginLoss(1.3, 3.2, 1e-5, True, False, True, 'mean')
        result = cri(input, positive, negative)
        """
    )
    obj.run(pytorch_code, ["result"])


# generated by validate_unittest autofix, based on test_case_8
def test_case_29():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, 5, 3], [0, 3, 2], [1, 4, 1]]).type(torch.float32)
        positive= torch.Tensor([[5, 1, 2], [3, 2, 1], [3, -1, 1]]).type(torch.float32)
        negative = torch.Tensor([[2, 1, -3], [1, 1, -1], [4, -2, 1]]).type(torch.float32)
        cri = torch.nn.TripletMarginLoss(margin=1.3, p=3.2, eps=1e-5, swap=True, size_average=False, reduce=True, reduction='mean')
        result = cri(input, positive, negative)
        """
    )
    obj.run(pytorch_code, ["result"])


# generated by validate_unittest autofix, based on test_case_8
def test_case_30():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, 5, 3], [0, 3, 2], [1, 4, 1]]).type(torch.float32)
        positive= torch.Tensor([[5, 1, 2], [3, 2, 1], [3, -1, 1]]).type(torch.float32)
        negative = torch.Tensor([[2, 1, -3], [1, 1, -1], [4, -2, 1]]).type(torch.float32)
        cri = torch.nn.TripletMarginLoss(reduction='mean', reduce=True, size_average=False, swap=True, eps=1e-5, p=3.2, margin=1.3)
        result = cri(input, positive, negative)
        """
    )
    obj.run(pytorch_code, ["result"])


# generated by validate_unittest autofix, based on test_case_9
def test_case_31():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, 5, 3], [0, 3, 2], [1, 4, 1]]).type(torch.float32)
        positive= torch.Tensor([[5, 1, 2], [3, 2, 1], [3, -1, 1]]).type(torch.float32)
        negative = torch.Tensor([[2, 1, -3], [1, 1, -1], [4, -2, 1]]).type(torch.float32)
        cri = torch.nn.TripletMarginLoss(1.3, 3.2, 1e-1, False, True, False, 'mean')
        result = cri(input, positive, negative)
        """
    )
    obj.run(pytorch_code, ["result"], rtol=1.0e-5, atol=1.0e-8)


# generated by validate_unittest autofix, based on test_case_9
def test_case_32():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, 5, 3], [0, 3, 2], [1, 4, 1]]).type(torch.float32)
        positive= torch.Tensor([[5, 1, 2], [3, 2, 1], [3, -1, 1]]).type(torch.float32)
        negative = torch.Tensor([[2, 1, -3], [1, 1, -1], [4, -2, 1]]).type(torch.float32)
        cri = torch.nn.TripletMarginLoss(margin=1.3, p=3.2, eps=1e-1, swap=False, size_average=True, reduce=False, reduction='mean')
        result = cri(input, positive, negative)
        """
    )
    obj.run(pytorch_code, ["result"], rtol=1.0e-5, atol=1.0e-8)


# generated by validate_unittest autofix, based on test_case_9
def test_case_33():
    pytorch_code = textwrap.dedent(
        """
        import torch
        input = torch.Tensor([[1, 5, 3], [0, 3, 2], [1, 4, 1]]).type(torch.float32)
        positive= torch.Tensor([[5, 1, 2], [3, 2, 1], [3, -1, 1]]).type(torch.float32)
        negative = torch.Tensor([[2, 1, -3], [1, 1, -1], [4, -2, 1]]).type(torch.float32)
        cri = torch.nn.TripletMarginLoss(reduction='mean', reduce=False, size_average=True, swap=False, eps=1e-1, p=3.2, margin=1.3)
        result = cri(input, positive, negative)
        """
    )
    obj.run(pytorch_code, ["result"], rtol=1.0e-5, atol=1.0e-8)
