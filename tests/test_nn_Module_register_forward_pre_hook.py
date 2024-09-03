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

obj = APIBase("torch.nn.Module.register_forward_pre_hook")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1., 2., 3.])
        class DoubleNum(torch.nn.Module):
            def forward(self, x):
                return 2*x
        DN = DoubleNum()
        DN.register_forward_pre_hook(lambda layer, input: input[0]*2)
        result = DN(x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1., 2., 3.])
        def forward_pre_hook(layer, input):
            return input[0] * 2
        class DoubleNum(torch.nn.Module):
            def forward(self, x):
                return 2*x
        DN = DoubleNum()
        DN.register_forward_pre_hook(forward_pre_hook)
        result = DN(x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        result = []
        class TestForHook(nn.Module):
            def __init__(self):
                super().__init__()

                self.linear_1 = nn.Linear(in_features=2, out_features=2)
            def forward(self, x):
                x1 = self.linear_1(x)
                return x, x1
        def hook(module, fea_in):
            result.append(1)

        net = TestForHook()
        net.register_forward_pre_hook(hook)
        a = torch.tensor([0.,0.])
        net(a)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        result = []
        class TestForHook(nn.Module):
            def __init__(self):
                super().__init__()

                self.linear_1 = nn.Linear(in_features=2, out_features=2)
            def forward(self, x):
                x1 = self.linear_1(x)
                return x, x1
        def hook(module, fea_in):
            result.append(1)

        net = TestForHook()
        net.register_forward_pre_hook(hook=hook, prepend=False, with_kwargs=False)
        a = torch.tensor([0.,0.])
        net(a)
        """
    )
    obj.run(
        pytorch_code, unsupport=True, reason="prepend and with_kwargs is not supported"
    )


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        result = []
        class TestForHook(nn.Module):
            def __init__(self):
                super().__init__()

                self.linear_1 = nn.Linear(in_features=2, out_features=2)
            def forward(self, x):
                x1 = self.linear_1(x)
                return x, x1
        def hook(module, fea_in):
            result.append(1)

        net = TestForHook()
        net.register_forward_pre_hook(hook=hook, with_kwargs=False, prepend=False)
        a = torch.tensor([0.,0.])
        net(a)
        """
    )
    obj.run(
        pytorch_code, unsupport=True, reason="prepend and with_kwargs is not supported"
    )
