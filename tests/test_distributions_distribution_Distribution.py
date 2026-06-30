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
import paddle
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
        if hasattr(pytorch_result, "batch_shape") and hasattr(
            pytorch_result, "event_shape"
        ):
            assert hasattr(paddle_result, "batch_shape") and hasattr(
                paddle_result, "event_shape"
            ), f"paddle result should be Distribution, got {type(paddle_result)}"
            assert pytorch_result.batch_shape == paddle_result.batch_shape
            assert pytorch_result.event_shape == paddle_result.event_shape
            if (
                hasattr(pytorch_result, "probs")
                and hasattr(paddle_result, "probs")
                and pytorch_result.probs is not None
            ):
                np.testing.assert_allclose(
                    pytorch_result.probs.detach().cpu().numpy(),
                    paddle_result.probs.numpy(),
                    rtol=rtol,
                    atol=atol,
                )
            if (
                hasattr(pytorch_result, "logits")
                and hasattr(paddle_result, "logits")
                and pytorch_result.logits is not None
            ):
                np.testing.assert_allclose(
                    pytorch_result.logits.detach().cpu().numpy(),
                    paddle_result.logits.numpy(),
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


class ResultAPIBase(APIBase):
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
        assert isinstance(paddle_result, paddle.distribution.Distribution)


obj = ResultAPIBase("torch.distributions.distribution.Distribution")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.distributions.distribution.Distribution()
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.distributions.distribution.Distribution(batch_shape=torch.Size([]), event_shape=torch.Size([]))
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.distributions.distribution.Distribution(event_shape=torch.Size([]), batch_shape=torch.Size([]), validate_args=None)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.distributions.distribution.Distribution(torch.Size([]), torch.Size([]), validate_args=None)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_6():
    """Expression argument test"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.distributions.distribution.Distribution(torch.tensor([1.0, 2.0, 3.0]) + torch.tensor([0.5, 0.5, 0.5]))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    """2D tensor test"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[1.4309, 1.2706], [-0.8562, 0.9796]])
        result = torch.distributions.distribution.Distribution(a)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    """3D tensor test"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]])
        result = torch.distributions.distribution.Distribution(a)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    """float64 dtype test"""
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([1.4309, 1.2706], dtype=torch.float64)
        result = torch.distributions.distribution.Distribution(a)
        """
    )
    obj.run(pytorch_code, ["result"])
