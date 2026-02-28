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

obj = APIBase("torch.Tensor.renorm")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[ 1.,  1.,  1.],
                            [ 2.,  2.,  2.],
                            [ 3.,  3.,  3.]])
        result = x.renorm(1, 0, 5)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[ 1.,  1.,  1.],
                            [ 2.,  2.,  2.],
                            [ 3.,  3.,  3.]])
        result = x.renorm(p=1, dim=0, maxnorm=5)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[ 1.,  1.,  1.],
                            [ 2.,  2.,  2.],
                            [ 3.,  3.,  3.]])
        out = torch.tensor([1., 3.])
        result = x.renorm(1, 0, 5)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[ 1.,  1.,  1.],
                            [ 2.,  2.,  2.],
                            [ 3.,  3.,  3.]])
        result = x.renorm(maxnorm=5, p=1, dim=0)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    """Test with expression argument"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[ 1.,  1.,  1.],
                            [ 2.,  2.,  2.],
                            [ 3.,  3.,  3.]])
        result = x.renorm(1 + 1, 0, 4 + 1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    """Test with mixed positional and keyword arguments"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[ 1.,  1.,  1.],
                            [ 2.,  2.,  2.],
                            [ 3.,  3.,  3.]])
        result = x.renorm(1, dim=0, maxnorm=5)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    """Test with different p value"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[ 1.,  1.,  1.],
                            [ 2.,  2.,  2.],
                            [ 3.,  3.,  3.]])
        result = x.renorm(p=2, dim=1, maxnorm=3)
        """
    )
    obj.run(pytorch_code, ["result"])
