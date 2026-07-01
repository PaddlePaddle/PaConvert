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

from dist_apibase import DistributionAPIBase

obj = DistributionAPIBase("torch.distributions.dirichlet.Dirichlet")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        m = torch.distributions.dirichlet.Dirichlet(torch.tensor([1.0, 1.0, 1.0]))
        result = m.sample([100])
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        m = torch.distributions.dirichlet.Dirichlet(concentration=torch.tensor([2.0, 2.0, 2.0]), validate_args=False)
        result = m.sample([100])
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        m = torch.distributions.dirichlet.Dirichlet(torch.tensor([1.0, 2.0, 3.0]), validate_args=False)
        result = m.sample([100])
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        m = torch.distributions.dirichlet.Dirichlet(concentration=torch.tensor([0.5, 0.5]))
        result = m.sample([100])
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_6():
    """Expression argument test"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.distributions.dirichlet.Dirichlet(torch.tensor([1.0, 2.0, 3.0]) + torch.tensor([0.5, 0.5, 0.5]))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    """3D tensor test"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]])
        result = torch.distributions.dirichlet.Dirichlet(a)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    """float64 dtype test"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([1.4309, 1.2706], dtype=torch.float64)
        result = torch.distributions.dirichlet.Dirichlet(a)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_11():
    """Keyword arguments out of order test"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        m = torch.distributions.dirichlet.Dirichlet(concentration=torch.tensor([1.0, 2.0, 3.0]), validate_args=False)
        result = m.sample([50])
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_12():
    """Mixed arguments test"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        m = torch.distributions.dirichlet.Dirichlet(torch.tensor([0.5, 1.0, 1.5]), validate_args=False)
        result = m.sample([30])
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)
