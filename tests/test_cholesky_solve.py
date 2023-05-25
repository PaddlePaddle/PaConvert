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

obj = APIBase("torch.cholesky_solve")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[ 2.4112, -0.7486,  1.4551],
                        [-0.7486,  1.3544,  0.1294],
                        [ 1.4551,  0.1294,  1.6724]])
        b = torch.tensor([[-0.6355,  0.9891],
                        [ 0.1974,  1.4706],
                        [-0.4115, -0.6225]])
        result = torch.cholesky_solve(b, x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[ 2.4112, -0.7486,  1.4551],
                        [-0.7486,  1.3544,  0.1294],
                        [ 1.4551,  0.1294,  1.6724]])
        b = torch.tensor([[-0.6355,  0.9891],
                        [ 0.1974,  1.4706],
                        [-0.4115, -0.6225]])
        result = torch.cholesky_solve(b, x, False)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[ 2.4112, -0.7486,  1.4551],
                        [-0.7486,  1.3544,  0.1294],
                        [ 1.4551,  0.1294,  1.6724]])
        b = torch.tensor([[-0.6355,  0.9891],
                        [ 0.1974,  1.4706],
                        [-0.4115, -0.6225]])
        result = torch.cholesky_solve(input=b, input2=x, upper=False)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[ 2.4112, -0.7486,  1.4551],
                        [-0.7486,  1.3544,  0.1294],
                        [ 1.4551,  0.1294,  1.6724]])
        b = torch.tensor([[-0.6355,  0.9891],
                        [ 0.1974,  1.4706],
                        [-0.4115, -0.6225]])
        out = torch.tensor([[-0.6355,  0.9891],
                        [ 0.1974,  1.4706],
                        [-0.4115, -0.6225]])
        result = torch.cholesky_solve(b, x, False, out=out)
        """
    )
    obj.run(pytorch_code, ["result", "out"])
