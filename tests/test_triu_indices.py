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

obj = APIBase("torch.triu_indices")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.triu_indices(3, 3)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.triu_indices(4, 3, -1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.triu_indices(4, 3, 1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.triu_indices(row=4, col=3, offset=-1, dtype=torch.int64)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.triu_indices(4, 3, -1, device=torch.device('cpu'), layout=torch.strided)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.triu_indices(row=4, col=3, offset=-1, dtype=torch.int64, device=torch.device('cpu'), layout=torch.strided)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.triu_indices(row=4, device=torch.device('cpu'), offset=-1, dtype=torch.int64, col=3, layout=torch.strided)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    """Rectangular matrix (more rows than cols)"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.triu_indices(row=5, col=3)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    """Rectangular matrix (more cols than rows)"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.triu_indices(row=2, col=5)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_10():
    """Large positive offset"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.triu_indices(row=5, col=5, offset=2)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_11():
    """1x1 matrix"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.triu_indices(row=1, col=1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_12():
    """Default dtype and device"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.triu_indices(3, 3, offset=0)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_13():
    """Expression argument test"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.triu_indices(2 + 1, 3, offset=-1)
        """
    )
    obj.run(pytorch_code, ["result"])
