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

obj = APIBase("torch.nn.functional.kl_div")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch.nn.functional as F
        import torch
        input = torch.arange(0, 15,dtype=torch.float32, requires_grad=True).reshape((3, 5))
        target = torch.arange(100, 160, 4, dtype=torch.float32, requires_grad=True).reshape((3, 5)) + 4
        result = F.kl_div(input, target)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch.nn.functional as F
        import torch
        input = torch.arange(0, 15,dtype=torch.float32, requires_grad=True).reshape((3, 5))
        target = torch.arange(100, 160, 4, dtype=torch.float32, requires_grad=True).reshape((3, 5)) + 4
        result = F.kl_div(input, target, True, True, "mean")
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch.nn.functional as F
        import torch
        input = torch.arange(0, 15,dtype=torch.float32, requires_grad=True).reshape((3, 5))
        target = torch.arange(100, 160, 4, dtype=torch.float32, requires_grad=True).reshape((3, 5)) + 4
        result = F.kl_div(input, target, False, True, "sum")
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch.nn.functional as F
        import torch
        input = torch.arange(0, 15,dtype=torch.float32, requires_grad=True).reshape((3, 5))
        target = torch.arange(100, 160, 4, dtype=torch.float32, requires_grad=True).reshape((3, 5)) + 4
        result = F.kl_div(input, target, None, True, "mean")
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch.nn.functional as F
        import torch
        input = torch.arange(0, 15,dtype=torch.float32, requires_grad=True).reshape((3, 5))
        target = torch.arange(100, 160, 4, dtype=torch.float32, requires_grad=True).reshape((3, 5)) + 4
        result = F.kl_div(input=input, target=target, size_average=True, reduce=False, reduction="mean")
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch.nn.functional as F
        import torch
        input = torch.arange(0, 15,dtype=torch.float32, requires_grad=True).reshape((3, 5))
        target = torch.arange(100, 160, 4, dtype=torch.float32, requires_grad=True).reshape((3, 5)) + 4
        result = F.kl_div(input, target, False, False)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch.nn.functional as F
        import torch
        input = torch.arange(0, 15,dtype=torch.float32, requires_grad=True).reshape((3, 5))
        target = torch.arange(100, 160, 4, dtype=torch.float32, requires_grad=True).reshape((3, 5)) + 4
        result = F.kl_div(input, target, None, False)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch.nn.functional as F
        import torch
        input = torch.arange(0, 15,dtype=torch.float32, requires_grad=True).reshape((3, 5))
        target = torch.arange(100, 160, 4, dtype=torch.float32, requires_grad=True).reshape((3, 5)) + 4
        result = F.kl_div(input, target, None, False, "sum")
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    pytorch_code = textwrap.dedent(
        """
        import torch.nn.functional as F
        import torch
        input = torch.arange(0, 15,dtype=torch.float32, requires_grad=True).reshape((3, 5))
        target = torch.arange(100, 160, 4, dtype=torch.float32, requires_grad=True).reshape((3, 5)) + 4
        result = F.kl_div(input=input, target=target,size_average=None, reduce=None, reduction="sum")
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_10():
    pytorch_code = textwrap.dedent(
        """
        import torch.nn.functional as F
        import torch
        input = torch.arange(0, 15,dtype=torch.float32, requires_grad=True).reshape((3, 5))
        target = torch.arange(100, 160, 4, dtype=torch.float32, requires_grad=True).reshape((3, 5)) + 4
        result = F.kl_div(input=input, target=target, size_average=None, reduce=None, reduction="sum", log_target=False)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_11():
    pytorch_code = textwrap.dedent(
        """
        import torch.nn.functional as F
        import torch
        input = torch.arange(0, 15,dtype=torch.float32, requires_grad=True).reshape((3, 5))
        target = torch.arange(100, 160, 4, dtype=torch.float32, requires_grad=True).reshape((3, 5)) + 4
        result = F.kl_div(input=input, reduce=None, reduction="sum", target=target, size_average=None)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_12():
    pytorch_code = textwrap.dedent(
        """
        import torch.nn.functional as F
        import torch
        input = torch.arange(0, 15,dtype=torch.float32, requires_grad=True).reshape((3, 5))
        target = torch.arange(100, 160, 4, dtype=torch.float32, requires_grad=True).reshape((3, 5)) + 4
        result = F.kl_div(input=input, target=target, size_average=None, reduce=None, reduction="sum", log_target=True)
        """
    )
    obj.run(pytorch_code, ["result"])
