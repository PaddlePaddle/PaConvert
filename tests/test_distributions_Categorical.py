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

import numpy as np
import pytest
from apibase import APIBase


class DistributionAPIBase(APIBase):
    """APIBase with custom compare logic for Distribution objects.

    Distribution objects are not Tensors and cannot be compared using the
    default Tensor-based compare logic. This compares batch_shape, event_shape,
    probs, and logits instead.
    """

    def compare(
        self,
        name,
        pytorch_result,
        paddle_result,
        check_value=True,
        check_shape=True,
        check_dtype=True,
        check_stop_gradient=True,
        rtol=1.0e-6,
        atol=0.0,
    ):
        if hasattr(pytorch_result, "batch_shape") and hasattr(
            pytorch_result, "event_shape"
        ):
            assert hasattr(paddle_result, "batch_shape") and hasattr(
                paddle_result, "event_shape"
            ), f"API ({name}): paddle result should be a Distribution object, but got {type(paddle_result)}"

            assert (
                pytorch_result.batch_shape == paddle_result.batch_shape
            ), f"API ({name}): batch_shape mismatch, torch is {pytorch_result.batch_shape}, paddle is {paddle_result.batch_shape}"
            assert (
                pytorch_result.event_shape == paddle_result.event_shape
            ), f"API ({name}): event_shape mismatch, torch is {pytorch_result.event_shape}, paddle is {paddle_result.event_shape}"

            for attr in ("probs", "logits"):
                p_src = getattr(pytorch_result, attr, None)
                d_src = getattr(paddle_result, attr, None)
                if p_src is not None and d_src is not None:
                    np.testing.assert_allclose(
                        p_src.detach().cpu().numpy(),
                        d_src.numpy(),
                        rtol=rtol,
                        atol=atol,
                    )
            return

        super().compare(
            name,
            pytorch_result,
            paddle_result,
            check_value,
            check_shape,
            check_dtype,
            check_stop_gradient,
            rtol,
            atol,
        )


obj = DistributionAPIBase("torch.distributions.Categorical")


def test_case_1():
    """Categorical with probs"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.distributions.Categorical(probs=torch.tensor([0.25, 0.25, 0.25, 0.25]))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    """Categorical with logits"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.distributions.Categorical(logits=torch.tensor([0.25, 0.25, 0.25, 0.25]))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    """Categorical with positional probs"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.distributions.Categorical(torch.tensor([0.3, 0.3, 0.4]))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    """Categorical with logits and validate_args"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.distributions.Categorical(logits=torch.tensor([0.25, 0.25, 0.25, 0.25]), validate_args=False)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    """Categorical with probs=None, logits provided"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.distributions.Categorical(probs=None, logits=torch.tensor([0.25, 0.25, 0.25, 0.25]))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    """Categorical with probs and logits=None"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.distributions.Categorical(probs=torch.tensor([0.25, 0.25, 0.25, 0.25]), logits=None)
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skip(
    reason="Paddle framework issue: PaConvert generates sample(shape=...) but paddle.compat.distributions.Categorical uses sample_shape= parameter"
)
def test_case_7():
    """Categorical with sample"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        m = torch.distributions.Categorical(logits=torch.tensor([0.25, 0.25, 0.25, 0.25]))
        result = m.sample([1])
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)
