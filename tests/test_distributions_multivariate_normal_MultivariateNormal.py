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

obj = APIBase("torch.distributions.multivariate_normal.MultivariateNormal")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        loc = torch.tensor([0.0, 0.0])
        covariance_matrix = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        m = torch.distributions.multivariate_normal.MultivariateNormal(loc=loc, covariance_matrix=covariance_matrix, validate_args=True)
        result = m.sample()
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        loc = torch.tensor([0.0, 0.0])
        covariance_matrix = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        m = torch.distributions.multivariate_normal.MultivariateNormal(loc, covariance_matrix)
        result = m.sample()
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        loc = torch.tensor([0.0, 0.0])
        scale_tril = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        m = torch.distributions.multivariate_normal.MultivariateNormal(loc, scale_tril=scale_tril)
        result = m.sample()
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        loc = torch.tensor([0.0, 0.0])
        scale_tril = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        m = torch.distributions.multivariate_normal.MultivariateNormal(loc, scale_tril=scale_tril)
        result = m.sample()
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        loc = torch.tensor([0.0, 0.0])
        precision_matrix = torch.tensor([
            [2.0, -1.0],
            [-1.0, 2.0]
        ])
        m = torch.distributions.multivariate_normal.MultivariateNormal(loc=loc, precision_matrix=precision_matrix)
        result = m.sample()
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        loc = torch.tensor([1.0, -0.5], dtype=torch.float32)
        covariance_matrix = torch.tensor([[2.0, 0.3], [0.3, 1.5]], dtype=torch.float32)
        m = torch.distributions.multivariate_normal.MultivariateNormal(
            loc=loc,
            covariance_matrix=covariance_matrix,
            validate_args=False,
        )
        mean = m.mean
        variance = m.variance
        entropy = m.entropy()
        log_prob = m.log_prob(torch.tensor([0.5, -1.0], dtype=torch.float32))
        """
    )
    obj.run(pytorch_code, ["mean", "variance", "entropy", "log_prob"])


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        loc = torch.tensor([[0.0, 1.0], [2.0, -1.0]], dtype=torch.float64)
        covariance_matrix = torch.tensor(
            [
                [[1.5, 0.2], [0.2, 0.8]],
                [[2.0, -0.4], [-0.4, 1.2]],
            ],
            dtype=torch.float64,
        )
        m = torch.distributions.multivariate_normal.MultivariateNormal(
            validate_args=None,
            covariance_matrix=covariance_matrix,
            loc=loc,
        )
        mean = m.mean
        variance = m.variance
        log_prob = m.log_prob(torch.tensor([[0.5, 0.5], [1.0, -0.5]], dtype=torch.float64))
        """
    )
    obj.run(pytorch_code, ["mean", "variance", "log_prob"])


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        args = (
            torch.tensor([0.5, -1.5], dtype=torch.float64),
            None,
            torch.tensor([[3.0, 0.5], [0.5, 2.0]], dtype=torch.float64),
            None,
            False,
        )
        m = torch.distributions.multivariate_normal.MultivariateNormal(*args)
        entropy = m.entropy()
        log_prob = m.log_prob(torch.tensor([0.0, -1.0], dtype=torch.float64))
        """
    )
    obj.run(pytorch_code, ["entropy", "log_prob"])


def test_case_9():
    pytorch_code = textwrap.dedent(
        """
        import torch
        loc = torch.tensor([[1.0, -2.0], [0.5, 3.0]], dtype=torch.float32)
        scale_tril = torch.tensor(
            [
                [[1.0, 0.0], [0.2, 0.8]],
                [[0.9, 0.0], [-0.1, 1.1]],
            ],
            dtype=torch.float32,
        )
        m = torch.distributions.multivariate_normal.MultivariateNormal(
            scale_tril=scale_tril,
            loc=loc,
            validate_args=True,
        )
        mean = m.mean
        log_prob = m.log_prob(torch.tensor([[0.5, -1.5], [1.0, 2.0]], dtype=torch.float32))
        """
    )
    obj.run(pytorch_code, ["mean", "log_prob"])
