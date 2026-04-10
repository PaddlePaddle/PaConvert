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

obj = APIBase("torch.unique")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor(
            [[2, 1, 3], [3, 0, 1], [2, 1, 3]])
        result = torch.unique(a)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor(
            [[2, 1, 3], [3, 0, 1], [2, 1, 3]])
        result = torch.unique(input=a, return_inverse=True, return_counts=True, dim=1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor(
            [[2, 1, 3], [3, 0, 1], [2, 1, 3]])
        dim = 1
        result = torch.unique(input=a, return_inverse=True, return_counts=False, dim=dim)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor(
            [[2, 1, 3], [3, 0, 1], [2, 1, 3]])
        dim = 1
        result = torch.unique(input=a, sorted=False, return_inverse=True, return_counts=False, dim=dim)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor(
            [[2, 1, 3], [3, 0, 1], [2, 1, 3]])
        dim = 1
        result = torch.unique(return_inverse=True, dim=dim, return_counts=False, input=a)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor(
            [[2, 1, 3], [3, 0, 1], [2, 1, 3]])
        dim = 1
        result = torch.unique(a, False, True, False, dim)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor(
            [[2, 1, 3], [3, 0, 1], [2, 1, 3]])
        dim = 1
        result = torch.unique(return_inverse=True, input=a, return_counts=False, dim=dim, sorted=False)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor(
            [[2, 1, 3], [3, 0, 1], [2, 1, 3]])
        dim = 1
        result = torch.unique(return_inverse=True, input=a, return_counts=False, dim=dim, sorted=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.arange(1001, 0, -1)
        a[1] = 1
        a[2] = 2
        dim = 0
        result = torch.unique(return_inverse=True, input=a, return_counts=False, dim=dim, sorted=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_10():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.arange(1001, 0, -1)
        a[1] = 1
        a[2] = 2
        dim = 0
        result = torch.unique(return_inverse=True, input=a, return_counts=False, dim=dim, sorted=False)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_11():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([1, 2, 1, 3, 2, 3, 3])
        result = torch.unique(a, return_counts=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_12():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[1, 2], [3, 4], [1, 2]])
        result = torch.unique(a, dim=0)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_13():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([1.5, 2.5, 1.5, 3.5])
        result = torch.unique(a, return_inverse=True, return_counts=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_14():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[[1, 2], [2, 3]], [[1, 2], [3, 4]]])
        result = torch.unique(a)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_15():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[1, 2, 3], [1, 2, 3], [4, 5, 6]])
        result = torch.unique(a, return_inverse=True, return_counts=True, dim=0)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_16():
    # variable args
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([1, 2, 1, 3, 2, 3, 3])
        args = (a,)
        result = torch.unique(*args)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_17():
    # kwargs dict unpacking
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([1, 2, 1, 3, 2, 3, 3])
        kwargs = {'input': a, 'return_inverse': True, 'return_counts': True}
        result = torch.unique(**kwargs)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_18():
    # negative dim
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[1, 2, 1], [3, 4, 3]])
        result = torch.unique(a, dim=-1)
        """
    )
    obj.run(pytorch_code, ["result"])
