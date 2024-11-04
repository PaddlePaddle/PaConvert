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

obj = APIBase("torch.Tensor.orgqr")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[-1.1280,  0.9012, -0.0190],
            [ 0.3699,  2.2133, -1.4792],
            [ 0.0308,  0.3361, -3.1761],
            [-0.0726,  0.8245, -0.3812]])
        input2 = torch.tensor([1.7497, 1.1156, 1.7462])
        result = x.orgqr(input2)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[-1.1280,  0.9012, -0.0190],
            [ 0.3699,  2.2133, -1.4792],
            [ 0.0308,  0.3361, -3.1761],
            [-0.0726,  0.8245, -0.3812]])
        input2 = torch.tensor([1.7497, 1.1156, 1.7462])
        result = x.orgqr(input2=input2)
        """
    )
    obj.run(pytorch_code, ["result"])
