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

obj = APIBase("torch.Tensor.nonzero")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.tensor([1, 1, 1, 0, 1]).nonzero()
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.tensor([[0.6, 0.0, 0.0, 0.0],
                                    [0.0, 0.4, 0.0, 0.0],
                                    [0.0, 0.0, 1.2, 0.0],
                                    [0.0, 0.0, 0.0,-0.4]]).nonzero()
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[0.6, 0.0, 0.0, 0.0],
                        [0.0, 0.4, 0.0, 0.0],
                        [0.0, 0.0, 1.2, 0.0],
                        [0.0, 0.0, 0.0,-0.4]])
        result = x.nonzero(as_tuple=True)
        """
    )
    obj.run(
        pytorch_code,
        [],
        reason="The return shape is inconsistent when as_tuple=True",
    )


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[0.6, 0.0, 0.0, 0.0],
                        [0.0, 0.4, 0.0, 0.0],
                        [0.0, 0.0, 1.2, 0.0],
                        [0.0, 0.0, 0.0,-0.4]])
        as_tuple = True
        result = x.nonzero(as_tuple=as_tuple)
        """
    )
    obj.run(
        pytorch_code,
        [],
        reason="The return shape is inconsistent when as_tuple=True",
    )


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[0.6, 0.0, 0.0, 0.0],
                        [0.0, 0.4, 0.0, 0.0],
                        [0.0, 0.0, 1.2, 0.0],
                        [0.0, 0.0, 0.0,-0.4]])
        result = x.nonzero(as_tuple=False)
        """
    )
    obj.run(pytorch_code, ["result"])
