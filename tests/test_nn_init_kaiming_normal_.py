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

obj = APIBase("torch.nn.init.kaiming_normal_")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        conv = torch.nn.Conv2d(4, 6, (3, 3))
        torch.nn.init.kaiming_normal_(conv.weight)
        result = conv.weight
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        conv = torch.nn.Conv2d(4, 6, (3, 3))
        torch.nn.init.kaiming_normal_(conv.weight)
        result = conv.weight
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        conv = torch.nn.Conv2d(3, 6, (3, 3))
        torch.nn.init.kaiming_normal_(tensor=conv.weight, a=0., mode='fan_in', nonlinearity='relu')
        result = conv.weight
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        conv = torch.nn.Conv2d(3, 6, (3, 3))
        torch.nn.init.kaiming_normal_(conv.weight, 1., 'fan_in', 'leaky_relu')
        result = conv.weight
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        conv = torch.nn.Conv2d(3, 6, (3, 3))
        torch.nn.init.kaiming_normal_(mode='fan_in', nonlinearity='leaky_relu', tensor=conv.weight, a=1.)
        result = conv.weight
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        conv = torch.nn.Conv2d(3, 6, (3, 3))
        torch.nn.init.kaiming_normal_(conv.weight, a=1., mode='fan_in')
        result = conv.weight
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        conv = torch.nn.Conv2d(3, 6, (3, 3))
        torch.nn.init.kaiming_normal_(conv.weight, a=1.)
        result = conv.weight
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        conv = torch.nn.Conv2d(3, 6, (3, 3))
        torch.nn.init.kaiming_normal_(conv.weight, a=1., mode='fan_out')
        result = conv.weight
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_9():
    pytorch_code = textwrap.dedent(
        """
        import torch
        conv = torch.nn.Conv2d(3, 6, (3, 3))
        torch.nn.init.kaiming_normal_(mode='fan_in', nonlinearity='leaky_relu', tensor=conv.weight, a=1., generator=None)
        result = conv.weight
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_10():
    for a in [0.0, 1.0, 0.3, 0.8]:
        for mode in ["fan_in", "fan_out"]:
            for nonlinearity in ["leaky_relu", "relu"]:
                for layer in [
                    "torch.nn.Linear(128, 256)",
                    "torch.nn.Conv2d(3, 6, (3, 3))",
                ]:
                    pytorch_code = textwrap.dedent(
                        f"""
                        import torch
                        linear = {layer}
                        torch.nn.init.kaiming_normal_(mode='{mode}', nonlinearity='{nonlinearity}', tensor=linear.weight, a={a})
                        result = linear.weight
                        """
                    )
                    obj.run(pytorch_code, ["result"], check_value=False)
