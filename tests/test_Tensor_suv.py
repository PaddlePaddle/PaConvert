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

obj = APIBase("torch.Tensor.svd")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[ 0.2364, -0.7752,  0.6372],
                        [ 1.7201,  0.7394, -0.0504],
                        [-0.3371, -1.0584,  0.5296],
                        [ 0.3550, -0.4022,  1.5569],
                        [ 0.2445, -0.0158,  1.1414]])
        result = x.svd()[1]
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[ 0.2364, -0.7752,  0.6372],
                        [ 1.7201,  0.7394, -0.0504],
                        [-0.3371, -1.0584,  0.5296],
                        [ 0.3550, -0.4022,  1.5569],
                        [ 0.2445, -0.0158,  1.1414]])
        result = x.svd(some=False)[1]
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[ 0.2364, -0.7752,  0.6372],
                        [ 1.7201,  0.7394, -0.0504],
                        [-0.3371, -1.0584,  0.5296],
                        [ 0.3550, -0.4022,  1.5569],
                        [ 0.2445, -0.0158,  1.1414]])
        result = x.svd(compute_uv=False)[1]
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[ 0.2364, -0.7752,  0.6372],
                        [ 1.7201,  0.7394, -0.0504],
                        [-0.3371, -1.0584,  0.5296],
                        [ 0.3550, -0.4022,  1.5569],
                        [ 0.2445, -0.0158,  1.1414]])
        result = x.svd(some=False, compute_uv=False)[1]
        """
    )
    obj.run(pytorch_code, ["result"])
