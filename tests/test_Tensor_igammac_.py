# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

obj = APIBase("torch.Tensor.igammac_")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5], dtype=torch.float)
        y = torch.tensor([0, 1, 10, 100, 1000], dtype=torch.float)
        result = x.igammac_(y)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5], dtype=torch.float)
        y = torch.tensor([0, 1, 10, 100, 1000], dtype=torch.float)
        result = x.igammac_(other=y)
        """
    )
    obj.run(pytorch_code, ["result"])
