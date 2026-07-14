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

obj = APIBase("torch.distributions.Normal")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        m = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        result = m.sample([1])
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        m = torch.distributions.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]), validate_args=False)
        result = m.sample([1])
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        m = torch.distributions.normal.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]), validate_args=False)
        result = m.sample([1])
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        m = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]), False)
        result = m.sample([1])
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        m = torch.distributions.normal.Normal(loc=torch.tensor([0.0]), validate_args=False, scale=torch.tensor([1.0]))
        result = m.sample([1])
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        loc = torch.tensor([[0.5, -1.5], [2.0, 3.5]], dtype=torch.float32)
        scale = torch.tensor([[1.0, 0.5], [2.0, 1.5]], dtype=torch.float32)
        m = torch.distributions.Normal(loc, scale)
        mean = m.mean
        variance = m.variance
        entropy = m.entropy()
        log_prob = m.log_prob(torch.tensor([[0.0, -1.0], [1.0, 4.0]], dtype=torch.float32))
        """
    )
    obj.run(pytorch_code, ["mean", "variance", "entropy", "log_prob"])


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        loc = torch.tensor([0.25, -0.75, 1.5], dtype=torch.float64)
        scale = torch.tensor([0.5, 1.25, 2.0], dtype=torch.float64)
        m = torch.distributions.normal.Normal(
            scale=scale,
            validate_args=True,
            loc=loc,
        )
        mean = m.mean
        variance = m.variance
        log_prob = m.log_prob(torch.tensor([0.0, -1.0, 2.0], dtype=torch.float64))
        """
    )
    obj.run(pytorch_code, ["mean", "variance", "log_prob"])


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        args = (
            torch.tensor([[1.0, -2.0], [3.0, 0.5]], dtype=torch.float64),
            torch.tensor([[0.75, 1.5], [2.5, 0.8]], dtype=torch.float64),
            None,
        )
        m = torch.distributions.Normal(*args)
        entropy = m.entropy()
        log_prob = m.log_prob(torch.tensor([[0.5, -1.5], [2.5, 1.0]], dtype=torch.float64))
        """
    )
    obj.run(pytorch_code, ["entropy", "log_prob"])


def test_case_9():
    pytorch_code = textwrap.dedent(
        """
        import torch
        loc = torch.tensor([[[0.2, -0.4], [1.1, 2.3]]], dtype=torch.float32)
        scale = torch.tensor([[[1.5, 0.7], [0.9, 2.1]]], dtype=torch.float32)
        m = torch.distributions.Normal(loc=loc, scale=scale, validate_args=None)
        result = m.log_prob(torch.tensor([[[0.0, -0.5], [1.0, 2.0]]], dtype=torch.float32))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_10():
    """Torch-style alias test"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        m = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        result = m.sample(sample_shape=torch.Size([2, 3]))
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)
