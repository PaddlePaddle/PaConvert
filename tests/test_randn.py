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

obj = APIBase("torch.randn")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.randn(3)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.randn(3, 5)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.randn((4, 4))
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.randn([4, 4])
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        shape = 3
        result = torch.randn(shape)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        shape = [4, 4]
        result = torch.randn(shape)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        shape = [4, 4]
        result = torch.randn(*shape)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.randn(size=[6, 6], dtype=torch.float64)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_9():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.randn([6, 6], dtype=torch.float64, requires_grad=True)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_10():
    pytorch_code = textwrap.dedent(
        """
        import torch
        flag = True
        result = torch.randn(6, 6, dtype=torch.float64, requires_grad=flag)
        """
    )
    obj.run(pytorch_code, ["result"], check_value=False)


def test_case_11():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = 3
        out = torch.tensor([2., 3.], dtype=torch.float64)
        result = torch.randn(size=(a, a), generator=None, out=out, dtype=torch.float64, layout=torch.strided, device=torch.device('cpu'), requires_grad=True, pin_memory=False)
        """
    )
    obj.run(pytorch_code, ["result", "out"], check_value=False)


def test_case_12():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = 3
        out = torch.tensor([2., 3.], dtype=torch.float64)
        result = torch.randn(pin_memory=False, dtype=torch.float64, out=out, layout=torch.strided, size=(a, a), generator=None, device=torch.device('cpu'), requires_grad=True)
        """
    )
    obj.run(pytorch_code, ["result", "out"], check_value=False)
