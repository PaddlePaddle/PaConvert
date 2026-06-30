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
from apibase import APIBase


class DistributionAPIBase(APIBase):
    """APIBase with custom compare logic for Distribution objects."""

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
        # Handle Distribution objects
        if hasattr(pytorch_result, "batch_shape") and hasattr(
            pytorch_result, "event_shape"
        ):
            assert hasattr(paddle_result, "batch_shape") and hasattr(
                paddle_result, "event_shape"
            ), f"API ({name}): paddle result should be a Distribution object, but got {type(paddle_result)}"

            assert (
                pytorch_result.batch_shape == paddle_result.batch_shape
            ), f"API ({name}): batch_shape mismatch"
            assert (
                pytorch_result.event_shape == paddle_result.event_shape
            ), f"API ({name}): event_shape mismatch"

            if hasattr(pytorch_result, "probs") and hasattr(paddle_result, "probs"):
                if pytorch_result.probs is not None and paddle_result.probs is not None:
                    pytorch_probs = pytorch_result.probs.detach().cpu().numpy()
                    paddle_probs = paddle_result.probs.numpy()
                    np.testing.assert_allclose(
                        pytorch_probs, paddle_probs, rtol=rtol, atol=atol
                    )
            if hasattr(pytorch_result, "logits") and hasattr(paddle_result, "logits"):
                if (
                    pytorch_result.logits is not None
                    and paddle_result.logits is not None
                ):
                    pytorch_logits = pytorch_result.logits.detach().cpu().numpy()
                    paddle_logits = paddle_result.logits.numpy()
                    np.testing.assert_allclose(
                        pytorch_logits, paddle_logits, rtol=rtol, atol=atol
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


obj = DistributionAPIBase("torch.distributions.bernoulli.Bernoulli")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        m = torch.distributions.bernoulli.Bernoulli(torch.tensor([0.3]))
        result = m.sample([100])
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        m = torch.distributions.bernoulli.Bernoulli(probs=torch.tensor([0.3]), logits=None)
        result = m.sample([100])
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        m = torch.distributions.bernoulli.Bernoulli(0.3, validate_args=False)
        result = m.sample([100])
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        m = torch.distributions.bernoulli.Bernoulli(probs=0.3, validate_args=False)
        result = m.sample([100])
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_8():
    """3D tensor test"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]])
        result = torch.distributions.bernoulli.Bernoulli(a)
        """
    )
    obj.run(pytorch_code, ["result"])
