# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

obj = APIBase("torch.nn.Module.__init__")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch

        class Demo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))

            def forward(self, x):
                return x * self.weight

        x = torch.tensor([0.5, 1.5, 2.5])
        result = Demo()(x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch

        class Demo(torch.nn.Module):
            def __init__(self, scale):
                super().__init__()
                self.scale = scale
                self.register_buffer("bias", torch.tensor([1.0, -1.0]))

            def forward(self, x):
                return x * self.scale + self.bias

        args = (torch.tensor([2.0, 3.0]),)
        x = torch.tensor([0.25, 0.75])
        result = Demo(*args)(x)
        """
    )
    obj.run(pytorch_code, ["result"])
