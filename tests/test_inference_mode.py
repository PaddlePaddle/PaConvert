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

obj = APIBase("torch.inference_mode", is_aux_api=True)


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.ones(1, 2, 3, requires_grad=True)
        @torch.inference_mode()
        def doubler(x):
            return x * 2
        result = (doubler(x).requires_grad, doubler(x))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.ones(1, 2, 3, requires_grad=True)
        @torch.inference_mode(True)
        def doubler(x):
            return x * 2
        result = (doubler(x).requires_grad, doubler(x))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.ones(1, 2, 3, requires_grad=True)
        @torch.inference_mode(False)
        def doubler(x):
            return x * 2
        result = (doubler(x).requires_grad, doubler(x))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.ones(1, 2, 3, requires_grad=True)
        with torch.inference_mode():
            y = x * 2
        result = (y.requires_grad, y)
        """
    )
    obj.run(pytorch_code, ["result"])
