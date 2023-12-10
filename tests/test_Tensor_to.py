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

obj = APIBase("torch.Tensor.to", is_aux_api=True)


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        cpu = torch.device('cpu')
        a =torch.ones(2, 3)
        c = torch.ones(2, 3, dtype= torch.float64, device=cpu)
        result = a.to(cpu, non_blocking=False, copy=False)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        cpu = torch.device('cpu')
        a =torch.ones(2, 3)
        result = a.to('cpu')
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        cpu = torch.device('cpu')
        a =torch.ones(2, 3)
        result = a.to(device = cpu, dtype = torch.float64)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        cpu = torch.device('cpu')
        a =torch.ones(2, 3)
        result = a.to(torch.float64)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a =torch.ones(2, 3)
        cpu = torch.device('cpu')
        result = a.to(dtype= torch.float64)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        cpu = torch.device('cpu')
        a =torch.ones(2, 3)
        c = torch.ones(2, 3, dtype= torch.float64, device=cpu)
        result = a.to(c)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        cpu = torch.device('cpu')
        a =torch.ones(2, 3)
        result = a.to(torch.half)
        """
    )
    obj.run(pytorch_code, ["result"])


# PR: https://github.com/PaddlePaddle/Paddle/pull/59857
def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        cpu = torch.device('cpu')
        a =torch.ones(2, 3)
        table =  a
        result = a.to(table.device)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    pytorch_code = textwrap.dedent(
        """
        import torch
        cpu = torch.device('cpu')
        a =torch.ones(2, 3)
        result = a.to(torch.float32)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_10():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.tensor([-1]).to(torch.bool)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_11():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a =torch.ones(2, 3)
        dtype = torch.float32
        result = a.to(dtype=dtype)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_12():
    pytorch_code = textwrap.dedent(
        """
        import torch
        cpu = torch.device('cpu')
        a =torch.ones(2, 3)
        result = a.to(torch.device('cpu'))
        """
    )
    obj.run(pytorch_code, ["result"])
