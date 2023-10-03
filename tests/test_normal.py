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

obj = APIBase("torch.normal")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.normal(torch.arange(1., 11.), torch.arange(1, 11))
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.normal(mean=0.5, std=torch.arange(1., 6.))
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.normal(mean=torch.arange(1., 6.))
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.normal(2, 3, size=(1, 4))
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        out = torch.empty(5)
        result = torch.normal(mean=torch.arange(1., 6.), out=out)
        """
    )
    obj.run(pytorch_code, ["result", "out"], check_value=False)


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        out = torch.empty(10, 10, dtype=torch.int32)
        result = torch.randint(0, 1, (10, 10), out=out, dtype=torch.int32, layout=torch.strided, device=torch.device('cpu'), pin_memory=False, requires_grad=False)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)
