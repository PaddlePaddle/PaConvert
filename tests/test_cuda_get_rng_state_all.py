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


class GetRngStateAllAPIBase(APIBase):
    def compare(
        self,
        name,
        pytorch_result,
        paddle_result,
        check_value=True,
        check_dtype=True,
        check_stop_gradient=True,
    ):
        if len(paddle_result) == 0:
            return True

        assert isinstance(paddle_result[0], paddle.fluid.libpaddle.GeneratorState)


obj = GetRngStateAllAPIBase("torch.cuda.get_rng_state_all")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.cuda.get_rng_state_all()
        """
    )
    obj.run(pytorch_code, ["result"])
