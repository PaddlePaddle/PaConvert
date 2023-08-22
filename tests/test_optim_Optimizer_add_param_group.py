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


class optimOptimizerAPIBase(APIBase):
    def compare(
        self,
        name,
        pytorch_result,
        paddle_result,
        check_value=True,
        check_dtype=True,
        check_stop_gradient=True,
        rtol=1.0e-6,
        atol=0.0,
    ):
        assert isinstance(paddle_result, paddle.optimizer.optimizer.Optimizer)


obj = optimOptimizerAPIBase("torch.optim.Optimizer.add_param_group")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.optim as optim
        w1 = torch.zeros(3, 3)
        w1.requires_grad = True
        w2 = torch.ones(3, 3)
        w2.requires_grad = True
        o = optim.Optimizer([w1], defaults={"learning_rate": 1.0})
        o.add_param_group({'params': w2})
        result0 = o.param_groups[0]["params"]
        result1 = o.param_groups[1]["params"]
        """
    )
    obj.run(pytorch_code, ["result0", "result1"])
