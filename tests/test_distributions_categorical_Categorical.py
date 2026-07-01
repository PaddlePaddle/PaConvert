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

import pytest
from dist_apibase import DistributionAPIBase

obj = DistributionAPIBase("torch.distributions.categorical.Categorical")


def _test_case_1():
    """Categorical with positional probs"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.distributions.categorical.Categorical(torch.tensor([0.3, 0.3, 0.4]))
        """
    )
    obj.run(pytorch_code, ["result"])


def _test_case_2():
    """Categorical with probs"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.distributions.categorical.Categorical(probs=torch.tensor([0.25, 0.25, 0.25, 0.25]))
        """
    )
    obj.run(pytorch_code, ["result"])


def _test_case_3():
    """Categorical with logits"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.distributions.categorical.Categorical(logits=torch.tensor([0.25, 0.25, 0.25, 0.25]))
        """
    )
    obj.run(pytorch_code, ["result"])


def _test_case_4():
    """Categorical with logits and validate_args"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.distributions.categorical.Categorical(logits=torch.tensor([0.25, 0.25, 0.25, 0.25]), validate_args=False)
        """
    )
    obj.run(pytorch_code, ["result"])


def _test_case_5():
    """Categorical with probs=None, logits provided"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.distributions.categorical.Categorical(probs=None, logits=torch.tensor([0.25, 0.25, 0.25, 0.25]))
        """
    )
    obj.run(pytorch_code, ["result"])


def _test_case_6():
    """Categorical with probs and logits=None"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.distributions.categorical.Categorical(probs=torch.tensor([0.25, 0.25, 0.25, 0.25]), logits=None)
        """
    )
    obj.run(pytorch_code, ["result"])


def _test_case_7():
    """Categorical with keyword arguments out of order"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.distributions.categorical.Categorical(validate_args=False, logits=torch.tensor([0.25, 0.25, 0.25, 0.25]))
        """
    )
    obj.run(pytorch_code, ["result"])


def _test_case_8():
    """3D tensor input"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]])
        result = torch.distributions.categorical.Categorical(a)
        """
    )
    obj.run(pytorch_code, ["result"])


def _test_case_9():
    """float64 dtype input"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([1.4309, 1.2706], dtype=torch.float64)
        result = torch.distributions.categorical.Categorical(a)
        """
    )
    obj.run(pytorch_code, ["result"])


def _test_case_10():
    """Expression as argument"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.distributions.categorical.Categorical(torch.tensor([1.0, 2.0, 3.0]) + torch.tensor([0.5, 0.5, 0.5]))
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skip(
    reason="Paddle framework issue: PaConvert generates sample(shape=...) but paddle.compat.distributions.categorical.Categorical uses sample_shape= parameter"
)
def _test_case_11():
    """Categorical with sample"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        m = torch.distributions.categorical.Categorical(logits=torch.tensor([0.25, 0.25, 0.25, 0.25]))
        result = m.sample([1])
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)
