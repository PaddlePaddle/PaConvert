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

obj = APIBase("torch.nn.functional.poisson_nll_loss")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = [[1.3192, 1.9915, 1.9674, 1.7151]]
        b = [[1.3492, 0.1915, 2.9434, 1.4151]]
        x1 = torch.tensor(a)
        x2 = torch.tensor(b)
        result = torch.nn.functional.poisson_nll_loss(x1, x2)
        """
    )
    obj.run(pytorch_code, ["x1", "x2", "result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = [[1.3192, 1.9915, 1.9674, 1.7151]]
        b = [[1.3492, 0.1915, 2.9434, 1.4151]]
        x1 = torch.tensor(a)
        x2 = torch.tensor(b)
        result = torch.nn.functional.poisson_nll_loss(x1, x2, log_input=False)
        """
    )
    obj.run(pytorch_code, ["x1", "x2", "result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = [[1.3192, 1.9915, 1.9674, 1.7151]]
        b = [[1.3492, 0.1915, 2.9434, 1.4151]]
        x1 = torch.tensor(a)
        x2 = torch.tensor(b)
        result = torch.nn.functional.poisson_nll_loss(x1, x2, full=True)
        """
    )
    obj.run(pytorch_code, ["x1", "x2", "result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = [[1.3192, 1.9915, 1.9674, 0]]
        b = [[1.3492, 0.1915, 2.9434, 1.4151]]
        x1 = torch.tensor(a)
        x2 = torch.tensor(b)
        result = torch.nn.functional.poisson_nll_loss(x1, x2, log_input=True, eps=1e-4)
        """
    )
    obj.run(pytorch_code, ["x1", "x2", "result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = [[1.3192, 1.9915, 1.9674, 0]]
        b = [[1.3492, 0.1915, 2.9434, 1.4151]]
        x1 = torch.tensor(a)
        x2 = torch.tensor(b)
        result = torch.nn.functional.poisson_nll_loss(x1, x2, reduction="sum")
        """
    )
    obj.run(pytorch_code, ["x1", "x2", "result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = [[1.3192, 1.9915, 1.9674, 0]]
        b = [[1.3492, 0.1915, 2.9434, 1.4151]]
        x1 = torch.tensor(a)
        x2 = torch.tensor(b)
        result = torch.nn.functional.poisson_nll_loss(x1, x2, reduction="none")
        """
    )
    obj.run(pytorch_code, ["x1", "x2", "result"])
