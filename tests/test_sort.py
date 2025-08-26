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

obj = APIBase("torch.sort")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[4, 9], [23, 2]])
        result = torch.sort(a)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[4, 9], [23, 2]])
        result = torch.sort(a, stable=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[4, 9], [23, 2]])
        result = torch.sort(a, 0, True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[4, 9], [23, 2]])
        result = torch.sort(input=a, dim=1, descending=True, stable=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        descending=False
        result = torch.sort(torch.tensor([[4, 9], [23, 2]]), dim=1, descending=descending, stable=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[4, 9], [23, 2]])
        out = (torch.tensor(a), torch.tensor(a))
        result = torch.sort(input=a, dim=1, descending=True, stable=True, out=out)
        """
    )
    obj.run(pytorch_code, ["result", "out"])


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[4, 9], [23, 2]])
        out = (torch.tensor(a), torch.tensor(a))
        result = torch.sort(a, dim=1, descending=True, stable=True, out=out)
        """
    )
    obj.run(pytorch_code, ["result", "out"])


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([ 90, 124, 101,  20,  67,  88,  22,  82, 116, 121,  8,  69, 32,
            100,  97,  25, 126, 114,  21,  90, 101,  34, 127, 105,  81,  72,
            28, 127, 127, 122,  33,  86])
        out = (torch.tensor(a), torch.tensor(a))
        result = torch.sort(descending=True, out=out, dim=0, input=a, stable=True)
        """
    )
    obj.run(pytorch_code, ["result", "out"])


def test_case_9():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[ 80,  23,  57],
            [ 69, 122,  39],
            [ 15,  36,  50],
            [121,   6,  65],
            [ 36,  35,  72],
            [ 18,  13,  15],
            [ 79,  86,  98],
            [107, 113,  30],
            [ 41,  53,  59],
            [ 23,  93, 116],
            [ 28,  32,  87],
            [ 89,  21,  20],
            [ 83,  24,  99],
            [  5,  15,  19],
            [ 92,  28,  48],
            [ 82, 117,  46]]).to(torch.int16)
        vals, inds = torch.sort(dim=-1, stable=False, input=a)
        """
    )
    obj.run(pytorch_code, ["vals", "inds"])


def test_case_10():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([ 90, 124, 101,  20,  67,  88,  22,  82, 116, 121,  8,  69, 32,
            100,  97,  25, 126, 114,  21,  90, 101,  34, 127, 105,  81,  72,
            28, 127, 127, 122,  33,  86], dtype=torch.float32)
        a.requires_grad = True
        b = a * a + a
        vals, inds = torch.sort(descending=True, dim=0, input=a, stable=True)
        vals.backward(torch.ones_like(vals))
        a.grad.requires_grad = False
        a_grad = a.grad
        """
    )
    obj.run(pytorch_code, ["a_grad"])


def test_case_11():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[90, 124, 101,  20],
            [67,  88,  22,  22],
            [116, 121,  8,  69],
            [32,  97,  97,  25],
            [126, 114,  21, 90],
            [101, 34, 127, 105],
            [81,  72, 28,   72],
            [127, 88,  22,  86]], dtype=torch.float64)
        a.requires_grad = True
        b = a * 2
        vals, inds = torch.sort(a, descending=False, dim=1, stable=False)
        vals.backward(torch.ones_like(vals))
        a.grad.requires_grad = False
        a_grad = a.grad
        """
    )
    obj.run(pytorch_code, ["a_grad"])
