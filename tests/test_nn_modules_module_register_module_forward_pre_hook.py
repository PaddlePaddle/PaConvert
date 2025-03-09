# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

obj = APIBase("torch.nn.modules.module.register_module_forward_pre_hook")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch

        def deterministic_pre_hook(module, args):
            modified_input = (args[0] + 0.5)
            return modified_input

        model = torch.nn.Linear(2, 2)
        with torch.no_grad():
            torch.nn.init.constant_(model.weight, 0.1)
            torch.nn.init.constant_(model.bias, 0.0)

        hook_handle = model.register_forward_pre_hook(deterministic_pre_hook)
        input = torch.tensor([[1.0, 2.0]], requires_grad=True)
        output = model(input)
        """
    )
    obj.run(pytorch_code, ["output"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch

        def deterministic_pre_hook(module, args):
            modified_input = (args[0] + 0.5)
            return modified_input

        model = torch.nn.Linear(2, 2)
        with torch.no_grad():
            torch.nn.init.constant_(model.weight, 0.1)
            torch.nn.init.constant_(model.bias, 0.0)

        hook_handle = model.register_forward_pre_hook(hook=deterministic_pre_hook)
        input = torch.tensor([[1.0, 2.0]], requires_grad=True)
        output = model(input)
        """
    )
    obj.run(pytorch_code, ["output"])
