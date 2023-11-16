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

obj = APIBase("torch.linalg.svd")


# Notice: In paddle, the cpu version and the gpu version symbols are different.
def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch

        A = torch.tensor(
            [[1.0, 2.0], [1.0, 3.0], [4.0, 6.0]]
        )
        u, s, v = torch.linalg.svd(A)
        # Symbols are different
        u = torch.abs(u)
        v = torch.abs(v)
        """
    )
    obj.run(pytorch_code, ["u", "s", "v"], atol=1e-5)


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch

        A = torch.tensor(
            [
                [0.2364, -0.7752, 0.6372],
                [1.7201, 0.7394, -0.0504],
                [-0.3371, -1.0584, 0.5296],
                [0.3550, -0.4022, 1.5569],
                [0.2445, -0.0158, 1.1414],
            ]
        )
        u, s, v = torch.linalg.svd(A=A, full_matrices=False)
        # Symbols are different
        u = torch.abs(u)
        v = torch.abs(v)
        """
    )
    obj.run(pytorch_code, ["u", "s", "v"], atol=1e-5)


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch

        A = torch.tensor(
            [
                [0.2364, -0.7752, 0.6372],
                [1.7201, 0.7394, -0.0504],
                [-0.3371, -1.0584, 0.5296],
                [0.3550, -0.4022, 1.5569],
                [0.2445, -0.0158, 1.1414],
            ]
        )
        u, s, v = torch.linalg.svd(driver=None, A=A, full_matrices=False)
        # Symbols are different
        u = torch.abs(u)
        v = torch.abs(v)
        """
    )
    obj.run(pytorch_code, ["u", "s", "v"], atol=1e-5)


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch

        A = torch.tensor(
            [
                [0.2364, -0.7752, 0.6372],
                [1.7201, 0.7394, -0.0504],
                [-0.3371, -1.0584, 0.5296],
                [0.3550, -0.4022, 1.5569],
                [0.2445, -0.0158, 1.1414],
            ]
        )
        out = [torch.tensor([]),torch.tensor([]),torch.tensor([])]
        u, s, v = torch.linalg.svd(A=A, full_matrices=True, driver=None, out=out)
        # Symbols are different
        u = torch.abs(u)
        v = torch.abs(v)
        out[0] = torch.abs(out[0])
        out[2] = torch.abs(out[2])
        """
    )
    obj.run(pytorch_code, ["u", "s", "v", "out"], atol=1e-5)


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch

        A = torch.tensor(
            [
                [0.2364, -0.7752, 0.6372],
                [1.7201, 0.7394, -0.0504],
                [-0.3371, -1.0584, 0.5296],
                [0.3550, -0.4022, 1.5569],
                [0.2445, -0.0158, 1.1414],
            ]
        )
        out = [torch.tensor([]),torch.tensor([]),torch.tensor([])]
        u, s, v = torch.linalg.svd(A, True, driver=None, out=out)
        # Symbols are different
        u = torch.abs(u)
        v = torch.abs(v)
        out[0] = torch.abs(out[0])
        out[2] = torch.abs(out[2])
        """
    )
    obj.run(pytorch_code, ["u", "s", "v", "out"], atol=1e-5)
