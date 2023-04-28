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

import os
import sys

sys.path.append(os.path.dirname(__file__) + "/../")

import textwrap

import paddle

from tests.apibase import APIBase


class GeneratorAPIBase(APIBase):
    def check(self, pytorch_result, paddle_result):
        if isinstance(paddle_result, paddle.fluid.libpaddle.Generator):
            return True
        return False


obj = GeneratorAPIBase("torch.Generator")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.Generator(device='cpu')
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.Generator()
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.Generator('cpu')
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        if torch.cuda.is_available():
            result = torch.Generator('cuda')
        else:
            result = torch.Generator('cpu')
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        if torch.cuda.is_available():
            result = torch.Generator(device='cuda')
        else:
            result = torch.Generator(device='cpu')
        """
    )
    obj.run(pytorch_code, ["result"])
