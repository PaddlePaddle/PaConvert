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

obj = APIBase("torch.distributions.Dirichlet")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        m = torch.distributions.Dirichlet(torch.tensor([0.3]))
        result = m.sample([100])
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        m = torch.distributions.Dirichlet(concentration=torch.tensor([0.3]), validate_args=False)
        result = m.sample([100])
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        m = torch.distributions.dirichlet.Dirichlet(torch.tensor([0.3]), validate_args=False)
        result = m.sample([100])
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        m = torch.distributions.dirichlet.Dirichlet(torch.tensor([0.3]), False)
        result = m.sample([100])
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        m = torch.distributions.Dirichlet(validate_args=False, concentration=torch.tensor([0.3]))
        result = m.sample([100])
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)
