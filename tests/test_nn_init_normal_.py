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

obj = APIBase("torch.nn.init.normal_")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        conv = torch.nn.Conv2d(4, 6, (3, 3))
        torch.nn.init.normal_(conv.weight)
        result = conv.weight
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        conv = torch.nn.Conv2d(4, 6, (3, 3))
        torch.nn.init.normal_(conv.weight, 0.2, 2.)
        result = conv.weight
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        conv = torch.nn.Conv2d(4, 6, (3, 3))
        torch.nn.init.normal_(conv.weight, mean=0.2, std=2.)
        result = conv.weight
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        conv = torch.nn.Conv2d(4, 6, (3, 3))
        torch.nn.init.normal_(conv.weight, std=2.)
        result = conv.weight
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        conv = torch.nn.Conv2d(4, 6, (3, 3))
        torch.nn.init.normal_(conv.weight, mean=0.2)
        result = conv.weight
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        conv = torch.nn.Conv2d(4, 6, (3, 3))
        torch.nn.init.normal_(tensor=conv.weight, mean=0.2, std=2.)
        result = conv.weight
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        conv = torch.nn.Conv2d(4, 6, (3, 3))
        torch.nn.init.normal_(mean=0.1, tensor=conv.weight)
        result = conv.weight
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        conv = torch.nn.Conv2d(4, 6, (3, 3))
        torch.nn.init.normal_(mean=0.1, std=1.5, tensor=conv.weight)
        result = conv.weight
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_9():
    pytorch_code = textwrap.dedent(
        """
        import torch
        conv = torch.nn.Conv2d(4, 6, (3, 3))
        torch.nn.init.normal_(mean=0.1, std=1.5, tensor=conv.weight, generator=None)
        result = conv.weight
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_10():
    pytorch_code = textwrap.dedent(
        """
        import torch
        linear = torch.nn.Linear(128, 256)
        torch.nn.init.normal_(mean=0.1, std=1.5, tensor=linear.weight)
        result = linear.weight
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_11():
    pytorch_code = textwrap.dedent(
        """
        import torch
        linear = torch.nn.Linear(128, 256)
        torch.nn.init.normal_(linear.weight, mean=0.1, std=1.5)
        result = linear.weight
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_12():
    pytorch_code = textwrap.dedent(
        """
        import torch
        linear = torch.nn.Linear(128, 256)
        torch.nn.init.normal_(linear.weight, 0.2, std=1.5)
        result = linear.weight
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_13():
    pytorch_code = textwrap.dedent(
        """
        import torch
        linear = torch.nn.Linear(128, 256)
        torch.nn.init.normal_(linear.weight, 0.2,)
        result = linear.weight
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_14():
    pytorch_code = textwrap.dedent(
        """
        import torch
        linear = torch.nn.Linear(128, 256)
        torch.nn.init.normal_(linear.weight, std = 1.2)
        result = linear.weight
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)
