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

obj = APIBase("torch.nn.Moudle.register_forward_hook")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1., 2., 3.])
        class DoubleNum(torch.nn.Module):
            def forward(self, x):
                return 2*x
        DN = DoubleNum()
        DN.register_forward_hook(lambda layer, input, output: output*2)
        result = DN(x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1., 2., 3.])
        def forward_post_hook(layer, input, output):
            return output * 2
        class DoubleNum(torch.nn.Module):
            def forward(self, x):
                return 2*x
        DN = DoubleNum()
        DN.register_forward_hook(forward_post_hook)
        result = DN(x)
        """
    )
    obj.run(pytorch_code, ["result"])
