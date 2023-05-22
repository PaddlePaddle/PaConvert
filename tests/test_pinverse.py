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

obj = APIBase("torch.pinverse")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[ 0.5495,  0.0979, -1.4092, -0.1128,  0.4132],
                            [-1.1143, -0.3662,  0.3042,  1.6374, -0.9294],
                            [-0.3269, -0.5745, -0.0382, -0.5922, -0.6759]])
        result = torch.pinverse(x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[ 0.5495,  0.0979, -1.4092, -0.1128,  0.4132],
                            [-1.1143, -0.3662,  0.3042,  1.6374, -0.9294],
                            [-0.3269, -0.5745, -0.0382, -0.5922, -0.6759]])
        result = torch.pinverse(x, rcond=1e-13)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[ 0.5495,  0.0979, -1.4092, -0.1128,  0.4132],
                            [-1.1143, -0.3662,  0.3042,  1.6374, -0.9294],
                            [-0.3269, -0.5745, -0.0382, -0.5922, -0.6759]])
        result = torch.pinverse(input=x, rcond=1e-13)
        """
    )
    obj.run(pytorch_code, ["result"])
