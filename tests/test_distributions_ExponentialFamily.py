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

import paddle
from apibase import APIBase


class ExponentialFamilyAPIBase(APIBase):
    def compare(
        self,
        name,
        pytorch_result,
        paddle_result,
        check_value=True,
        check_dtype=True,
        check_stop_gradient=True,
    ):
        if isinstance(paddle_result, paddle.distribution.ExponentialFamily):
            return True
        return False


obj = ExponentialFamilyAPIBase("torch.distributions.ExponentialFamily")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.distributions.ExponentialFamily()
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.distributions.ExponentialFamily(batch_shape=torch.Size([1]), event_shape=torch.Size([2]))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.distributions.ExponentialFamily(batch_shape=torch.Size([1]), event_shape=torch.Size([2]), validate_args=False)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.distributions.exp_family.ExponentialFamily(batch_shape=torch.Size([1]), event_shape=torch.Size([2]), validate_args=False)
        """
    )
    obj.run(pytorch_code, ["result"])
