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


class optimOptimizerAPIBase(APIBase):
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
        if isinstance(paddle_result, np.ndarray):
            np.testing.assert_array_equal(paddle_result, pytorch_result)
        else:
            assert paddle_result == pytorch_result


obj = optimOptimizerAPIBase("torch.optim.Optimizer.step")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn

        theta = torch.tensor([1.0,1.0], requires_grad=True)
        optim = torch.optim.Optimizer([theta], defaults={"learning_rate": 1.0})
        result = type(optim.step)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        theta = torch.tensor([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0], requires_grad=True)
        l = torch.nn.Linear(10, 1)
        torch.nn.init.constant_(l.weight, 3.0)
        torch.nn.init.constant_(l.bias, 4.0)
        optim = torch.optim.SGD(l.parameters(), lr=1.0)
        z = l(theta)
        z.backward()
        optim.step()
        weight = l.weight.data.numpy(force=True)
        bias = l.bias.data.numpy(force=True)
        """
    )
    obj.run(pytorch_code, ["weight", "bias"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn

        theta = torch.tensor([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0], requires_grad=True)
        l = torch.nn.Linear(10, 1)
        torch.nn.init.constant_(l.weight, 3.0)
        torch.nn.init.constant_(l.bias, 4.0)
        optim = torch.optim.SGD(l.parameters(), lr=0.01)

        def closure():
            optim.zero_grad()
            z = l(theta)
            loss = z.pow(2).sum()
            loss.backward()
            return loss

        loss_value = optim.step(closure=closure)
        result = loss_value.numpy(force=True)
        """
    )
    obj.run(pytorch_code, ["result"])
