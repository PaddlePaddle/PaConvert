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

obj = APIBase("torch.Tensor.requires_grad")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        data = torch.tensor([23.,32., 43.])
        result = 1
        if not data.requires_grad:
            result = 2
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        data = torch.tensor([23.,32., 43.])
        result = data.requires_grad
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        data = torch.tensor([23.,32., 43.])
        data.requires_grad = False
        result = data.requires_grad
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        data = torch.tensor([23.,32., 43.])
        data = torch.tensor([23.,32., 43.], requires_grad=data.requires_grad)
        result = data.requires_grad
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        data = torch.tensor([23.,32., 43.])
        result = data.requires_grad == False
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        data = torch.tensor([23.,32., 43.])
        result = not data.requires_grad
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        data = torch.tensor([23.,32., 43.])
        result = '{} , {}'.format("1", str(data.requires_grad))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        data = torch.tensor([23.,32., 43.])
        def test():
            return True
        data.requires_grad = test()
        result = data.requires_grad
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    pytorch_code = textwrap.dedent(
        """
        import torch
        data = torch.tensor([23.,32., 43.])
        z = (True, False, True)
        a, data.requires_grad, c = z
        result = data.requires_grad
        """
    )
    obj.run(pytorch_code, ["result"])
