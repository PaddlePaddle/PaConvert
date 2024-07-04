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

obj = APIBase("torch.dsplit")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.arange(16.0).reshape(2, 2, 4)
        result = torch.dsplit(a, 2)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.arange(16.0).reshape(2, 2, 4)
        result = torch.dsplit(a, [2,2])
        if len(result) > 2:
            result = (result[0], result[2])
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.arange(12).reshape(3, 2, 2)
        result = torch.dsplit(a, indices=[1,1])
        if len(result) > 2:
            result = (result[0], result[2])
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.arange(16.0).reshape(2, 2, 4)
        result = torch.dsplit(a,sections=2)
        """
    )
    obj.run(pytorch_code, ["result"])
