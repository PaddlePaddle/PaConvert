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
#

import textwrap

from apibase import APIBase
from unary_inplace_test_utils import register_standard_unary_inplace_tests

obj = APIBase("torch.Tensor.asin_")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.tensor([0.34, -0.56, 0.73]).asin_()
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.tensor([0.34, -0.56, 0.73])
        result = a.asin_()
        """
    )
    obj.run(pytorch_code, ["a", "result"])


register_standard_unary_inplace_tests(
    globals(),
    obj,
    "asin_",
    "[[-0.9, -0.25, 0.25], [0.9, -0.5, 0.5]]",
    "[[[-0.9, -0.25], [0.25, 0.9]], [[-0.5, 0.5], [0.1, -0.1]]]",
)
