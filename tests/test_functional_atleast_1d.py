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

obj = APIBase("torch.functional.atleast_1d")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.functional.atleast_1d(torch.tensor(123, dtype=torch.int32))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        y = torch.tensor([-1, -2, 3])
        result = torch.functional.atleast_1d((torch.tensor(123, dtype=torch.int32), y))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.functional.atleast_1d([torch.tensor([-1, -2, 3]), torch.tensor([-1, -2, 3])])
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor(1)
        result = torch.functional.atleast_1d(x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor(1)
        y = torch.tensor(2)
        z = torch.tensor(3)
        result = torch.functional.atleast_1d((x, y, z))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor(1)
        y = torch.tensor(2)
        z = torch.tensor(3)
        result = torch.functional.atleast_1d([x, y, z])
        """
    )
    obj.run(pytorch_code, ["result"])


# TODO: fix torch.atleast bug, which not support input list/tuple
def _test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor(1)
        y = torch.tensor(2)
        z = torch.tensor(3)
        tensors = [x, y, z]
        result = torch.functional.atleast_1d(tensors)
        """
    )
    obj.run(pytorch_code, ["result"])


# TODO: fix torch.atleast bug, which not support input list/tuple
def _test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor(1)
        y = torch.tensor(2)
        z = torch.tensor(3)
        tensors = (x, y, z)
        result = torch.functional.atleast_1d(tensors)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor(1)
        y = torch.tensor(2)
        z = torch.tensor(3)
        result = torch.functional.atleast_1d(x, y, z)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_10():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor(1)
        y = torch.tensor(2)
        z = torch.tensor(3)
        tensors = (x, y, z)
        result = torch.functional.atleast_1d(*tensors)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_11():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor(1)
        y = torch.tensor(2)
        z = torch.tensor(3)
        tensors = [x, y, z]
        result = torch.functional.atleast_1d(*tensors)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_12():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.functional.atleast_1d()
        """
    )
    obj.run(pytorch_code, ["result"])
