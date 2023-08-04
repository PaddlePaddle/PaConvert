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

obj = APIBase("torch.cov")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[0.95311481, 0.56955051, 0.50434124],
            [0.73109186, 0.35652584, 0.86189222]])
        result = torch.cov(a)
        """
    )
    obj.run(pytorch_code, ["result"], rtol=1.0e-5, atol=1.0e-8)


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[0.95311481, 0.56955051, 0.50434124],
            [0.73109186, 0.35652584, 0.86189222]])
        result = torch.cov(input=a, correction=0)
        """
    )
    obj.run(pytorch_code, ["result"], rtol=1.0e-5, atol=1.0e-8)


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[0.95311481, 0.56955051, 0.50434124],
            [0.73109186, 0.35652584, 0.86189222]])
        fw = torch.tensor([1, 6, 9])
        aw = torch.tensor([0.4282, 0.0255, 0.4144])
        result = torch.cov(input=a, correction=0, fweights=fw, aweights=aw)
        """
    )
    obj.run(pytorch_code, ["result"], rtol=1.0e-5, atol=1.0e-8)


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[0.95311481, 0.56955051, 0.50434124],
            [0.73109186, 0.35652584, 0.86189222]])
        fw = torch.tensor([1, 6, 9])
        aw = torch.tensor([0.4282, 0.0255, 0.4144])
        result = torch.cov(a, correction=0, fweights=fw, aweights=aw)
        """
    )
    obj.run(pytorch_code, ["result"], rtol=1.0e-5, atol=1.0e-8)


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[0., 1., 2.],
            [2., 1., 0.]])
        fw = torch.tensor([1, 6, 9])
        result = torch.cov(a, correction=1, fweights=fw)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[0.95311481, 0.56955051, 0.50434124],
            [0.73109186, 0.35652584, 0.86189222]])
        aw = torch.tensor([0.4282, 0.0255, 0.4144])
        result = torch.cov(a, correction=1, aweights=aw)
        """
    )
    obj.run(pytorch_code, ["result"], rtol=1.0e-5, atol=1.0e-8)
