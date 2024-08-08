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

obj = APIBase("torch.attribute")


# Attribute
def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1.0,2.0])
        result = x.T.requires_grad
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1.0,2.0])
        result = x.data.requires_grad
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([4, 6], dtype=torch.cfloat)
        result = x.real.requires_grad
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([4, 6], dtype=torch.cfloat)
        result = x.imag.requires_grad
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        linear = torch.nn.Linear(4, 4)
        result = linear.weight.requires_grad
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        linear = torch.nn.Linear(4, 4)
        result = linear.bias.requires_grad
        """
    )
    obj.run(pytorch_code, ["result"])


# call
def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1.0,2.0])
        result = x.T.abs()
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1.0,2.0])
        result = x.data.abs()
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([4, 6], dtype=torch.cfloat)
        result = x.real.abs()
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_10():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([4, 6], dtype=torch.cfloat)
        result = x.imag.abs()
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_11():
    pytorch_code = textwrap.dedent(
        """
        import torch
        linear = torch.nn.Linear(4, 4)
        result = linear.weight.abs()
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_12():
    pytorch_code = textwrap.dedent(
        """
        import torch
        linear = torch.nn.Linear(4, 4)
        result = linear.bias.abs()
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


# recursive call
def test_case_13():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1.0,2.0])
        result = torch.tan(x).T.abs()
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_14():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1.0,2.0])
        result = torch.tan(x).data.abs()
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_15():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([4, 6], dtype=torch.cfloat)
        result = torch.tan(x).real.abs()
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_16():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([4, 6], dtype=torch.cfloat)
        result = torch.tan(x).imag.abs()
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_17():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.nn.Linear(4, 4).weight.abs()
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_18():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.nn.Linear(4, 4).bias.abs()
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


# random combination
def test_case_19():
    pytorch_code = textwrap.dedent(
        """
        import torch
        linear = torch.nn.Linear(4, 4)
        result = linear.weight.data.T
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False, check_stop_gradient=False)


def test_case_20():
    pytorch_code = textwrap.dedent(
        """
        import torch
        linear = torch.nn.Linear(4, 4)
        result = linear.weight.T.data
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False, check_stop_gradient=False)


def test_case_21():
    pytorch_code = textwrap.dedent(
        """
        import torch
        linear = torch.nn.Linear(4, 4)
        result = linear.weight.T.T.data.T
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False, check_stop_gradient=False)
