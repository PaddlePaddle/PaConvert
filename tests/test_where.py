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

obj = APIBase("torch.where")

# The type of data we are trying to retrieve does not match the type of data currently contained in the container


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[0.9383, -0.1983, 3.2, -1.2]])
        y = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        result = torch.where(x>0, x, y)
        """
    )
    obj.run(pytorch_code, ["result"])


# paddle.where not support type promote and x/y must have same dtype
def _test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[0.9383, -0.1983, 3.2, -1.2]])
        result = torch.where(x>0, x, torch.tensor(90))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[3, 0], [4.8, 9.2]])
        result = torch.where(x)
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        unsupport=True,
        reason="The return shape is inconsistent when only pass condition param",
    )
