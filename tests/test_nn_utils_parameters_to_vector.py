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


class API(APIBase):
    def check(self, pytorch_result, paddle_result):
        torch_numpy, paddle_numpy = (
            pytorch_result.detach().numpy(),
            paddle_result.numpy(),
        )
        if torch_numpy.shape != paddle_numpy.shape:
            return False
        if str(pytorch_result.dtype)[6:] != str(paddle_result.dtype)[7:]:
            return False
        return True


obj = API("torch.nn.utils.parameters_to_vector")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        linear = torch.nn.Linear(10, 15)
        result = torch.nn.utils.parameters_to_vector(linear.parameters())
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        linear = torch.nn.Linear(10, 15)
        result = torch.nn.utils.parameters_to_vector(parameters=linear.parameters())
        """
    )
    obj.run(pytorch_code, ["result"])
