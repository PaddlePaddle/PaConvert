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

obj = APIBase("torch.Tensor.corrcoef")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[ 0.7308,  1.0060,  0.5270,  1.4516],
                        [-0.1383,  1.5706,  0.4724,  0.4141],
                        [ 0.1193,  0.2829,  0.9037,  0.3957],
                        [-0.8202, -0.6474, -0.1631, -0.6543]])
        result = x.corrcoef()
        """
    )
    obj.run(pytorch_code, ["result"], rtol=1.0e-5, atol=1.0e-8)


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[-0.1533,  2.3020, -0.1771,  0.5928],
                          [ 0.4338, -0.6537,  0.2296,  0.5946],
                          [-0.4932,  1.8386, -0.1039,  1.0440],
                          [ 0.1735, -0.8303, -0.3821, -0.4384],
                          [-0.1533,  2.3020, -0.1771,  0.5928],
                          [ 0.4338, -0.6537,  0.2296,  0.5946],
                          [-0.4932,  1.8386, -0.1039,  1.0440],
                          [ 0.1735, -0.8303, -0.3821, -0.4384]])
        result = x.corrcoef()
        """
    )
    obj.run(pytorch_code, ["result"], rtol=1.0e-5, atol=1.0e-8)
