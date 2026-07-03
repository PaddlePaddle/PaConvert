# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

obj = APIBase("torch.nn.modules.activation.ReLU")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        model = torch.nn.modules.activation.ReLU()
        result = model(torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]))
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        model = torch.nn.modules.activation.ReLU()
        result = model(torch.tensor([-1.0, 0.0, 2.0, -3.0, 4.0]))
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        model = torch.nn.modules.activation.ReLU()
        result = model(torch.randn(3, 4))
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        model = torch.nn.modules.activation.ReLU()
        result = model(torch.randn(2, 3, 4))
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        model = torch.nn.modules.activation.ReLU()
        result = model(torch.tensor([-1.5, 0.0, 3.0], dtype=torch.float64))
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_9():
    pytorch_code = textwrap.dedent(
        """
        import torch
        model = torch.nn.modules.activation.ReLU(inplace=False)
        result = model(torch.randn(2, 5))
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_10():
    pytorch_code = textwrap.dedent(
        """
        import torch
        model = torch.nn.modules.activation.ReLU()
        result = model(torch.tensor([[1.0, -2.0], [-3.0, 4.0]]))
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)
