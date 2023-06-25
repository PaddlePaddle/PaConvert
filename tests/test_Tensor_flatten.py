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

obj = APIBase("torch.Tensor.flatten")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[[3.4742,  0.5466, -0.8008, -0.9079], [3.4742,  0.5466, -0.8008, -0.9079]]])
        result = x.flatten()
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[[3.4742,  0.5466, -0.8008, -0.9079], [3.4742,  0.5466, -0.8008, -0.9079]]])
        result = x.flatten(1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[[3.4742,  0.5466, -0.8008, -0.9079], [3.4742,  0.5466, -0.8008, -0.9079]]])
        result = x.flatten(1, 2)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[[3.4742,  0.5466, -0.8008, -0.9079], [3.4742,  0.5466, -0.8008, -0.9079]]])
        result = x.flatten(start_dim=1, end_dim=-1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[[3.4742,  0.5466, -0.8008, -0.9079], [3.4742,  0.5466, -0.8008, -0.9079]]])
        result = x.flatten(start_dim=1, end_dim=2)
        """
    )
    obj.run(pytorch_code, ["result"])
