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

obj = APIBase("torch.linalg.slogdet")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[[ 0.9254, -0.6213],
            [-0.5787,  1.6843]],

            [[ 0.3242, -0.9665],
            [ 0.4539, -0.0887]],

            [[ 1.1336, -0.4025],
            [-0.7089,  0.9032]]])
        result = torch.linalg.slogdet(a)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[[ 0.9254, -0.6213],
            [-0.5787,  1.6843]],

            [[ 0.3242, -0.9665],
            [ 0.4539, -0.0887]],

            [[ 1.1336, -0.4025],
            [-0.7089,  0.9032]]])
        result = torch.linalg.slogdet(a)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[ 0.7308,  1.0060,  0.5270,  1.4516],
                        [-0.1383,  1.5706,  0.4724,  0.4141],
                        [ 0.1193,  0.2829,  0.9037,  0.3957],
                        [-0.8202, -0.6474, -0.1631, -0.6543]])
        out = (torch.tensor(0.), torch.tensor(0.))
        result = torch.linalg.slogdet(A=x, out=out)
        """
    )
    obj.run(pytorch_code, ["result", "out"])
