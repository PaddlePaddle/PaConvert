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

obj = APIBase("torch.aminmax")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
        result = torch.aminmax(t)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
        result = torch.aminmax(t, dim=-1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
        result = torch.aminmax(t, dim=-1, keepdim=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
        out = tuple([torch.tensor(-1.), torch.tensor(-1.)])
        result = torch.aminmax(t, out=out)
        """
    )
    obj.run(pytorch_code, ["out"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        dim, keepdim = 1, False
        out = tuple([torch.tensor([-1, -1]), torch.tensor([-1, -1])])
        torch.aminmax(input=torch.tensor([[1, 2, 3], [3, 4, 6]]), dim=dim, keepdim=keepdim, out=out)
        """
    )
    obj.run(pytorch_code, ["out"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        dim, keepdim = 1, False
        out = tuple([torch.tensor([-1, -1]), torch.tensor([-1, -1])])
        torch.aminmax(dim=dim, out=out, keepdim=keepdim, input=torch.tensor([[1, 2, 3], [3, 4, 6]]))
        """
    )
    obj.run(pytorch_code, ["out"])


def test_case_7():
    """1D tensor input"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        t = torch.tensor([5.0, 1.0, 3.0, 9.0, 2.0])
        result = torch.aminmax(t)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    """3D tensor with dim=0"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        t = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        result = torch.aminmax(t, dim=0)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    """3D tensor with dim=1 and keepdim=True"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        t = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        result = torch.aminmax(t, dim=1, keepdim=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_10():
    """float64 dtype"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        t = torch.tensor([[1.5, 2.3, 3.7], [4.1, 5.9, 6.2]], dtype=torch.float64)
        result = torch.aminmax(t, dim=1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_11():
    """explicit keepdim=False"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
        result = torch.aminmax(t, dim=0, keepdim=False)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_12():
    """input as keyword argument only"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.aminmax(input=torch.tensor([3.0, 1.0, 4.0, 1.0, 5.0]))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_13():
    """kwargs unpacking"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        t = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        kwargs = {"dim": 0, "keepdim": True}
        result = torch.aminmax(t, **kwargs)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_14():
    """expression as dim argument"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        t = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = torch.aminmax(t, dim=2 - 1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_15():
    """dim with out parameter and keepdim"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        t = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        out = tuple([torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0])])
        result = torch.aminmax(t, dim=1, keepdim=False, out=out)
        """
    )
    obj.run(pytorch_code, ["out", "result"])
