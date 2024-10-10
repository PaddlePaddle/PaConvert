# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

obj = APIBase("torch._assert")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1., 2., 3.])
        torch._assert(x==x, "not equal")
        """
    )
    obj.run(pytorch_code)


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1., 2., 3.])
        y = x + 1
        torch._assert(condition=(x==y), message="not equal")
        """
    )
    obj.run(pytorch_code)


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1., 2., float('nan')])
        y = x
        torch._assert((x==y), message="not equal")
        """
    )
    obj.run(pytorch_code)


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1., 2., float('nan')])
        y = x
        torch._assert(message="not equal", condition=(x==y))
        """
    )
    obj.run(pytorch_code)


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1., 2., float('nan')])
        y = x
        torch._assert(message="not equal", (x==y))
        """
    )
    obj.run(pytorch_code)
