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

obj = APIBase("torch.nn.modules.module.register_module_forward_hook")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch

        def deterministic_post_hook(module, input, output):
            modified_output = output + 0.5
            return modified_output

        model = torch.nn.Linear(2, 2)
        with torch.no_grad():
            model.weight.data = torch.ones_like(model.weight) * 0.1
            model.bias.data = torch.zeros_like(model.bias)


        hook_handle = model.register_forward_hook(deterministic_post_hook)
        input_tensor = torch.tensor([[1.0, 2.0]], requires_grad=True)
        output = model(input_tensor)

        hook_handle.remove()
        output = model(input_tensor)
        """
    )
    obj.run(pytorch_code, ["output"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch

        def deterministic_post_hook(module, input, output):
            modified_output = output + 0.5
            return modified_output

        model = torch.nn.Linear(2, 2)
        with torch.no_grad():
            model.weight.data = torch.ones_like(model.weight) * 0.1
            model.bias.data = torch.zeros_like(model.bias)


        hook_handle = model.register_forward_hook(hook = deterministic_post_hook)
        input_tensor = torch.tensor([[1.0, 2.0]], requires_grad=True)
        output = model(input_tensor)

        hook_handle.remove()
        output = model(input_tensor)
        """
    )
    obj.run(pytorch_code, ["output"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch

        def deterministic_post_hook(module, input, output):
            modified_output = output + 0.5
            return modified_output

        model = torch.nn.Linear(2, 2)
        with torch.no_grad():
            model.weight.data = torch.ones_like(model.weight) * 0.1
            model.bias.data = torch.zeros_like(model.bias)


        hook_handle = model.register_forward_hook(hook = deterministic_post_hook, prepend=True, with_kwargs=True, always_call=True)
        input_tensor = torch.tensor([[1.0, 2.0]], requires_grad=True)
        output = model(input_tensor)

        hook_handle.remove()
        output = model(input_tensor)
        """
    )
    obj.run(
        pytorch_code,
        ["output"],
        unsupport=True,
        reason='paddle do not support arg "prepend"',
    )


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch

        def deterministic_post_hook(module, input, output):
            modified_output = output + 0.5
            return modified_output

        model = torch.nn.Linear(2, 2)
        with torch.no_grad():
            model.weight.data = torch.ones_like(model.weight) * 0.1
            model.bias.data = torch.zeros_like(model.bias)


        hook_handle = model.register_forward_hook(hook = deterministic_post_hook, with_kwargs=True, prepend=True, always_call=True)
        input_tensor = torch.tensor([[1.0, 2.0]], requires_grad=True)
        output = model(input_tensor)

        hook_handle.remove()
        output = model(input_tensor)
        """
    )
    obj.run(
        pytorch_code,
        ["output"],
        unsupport=True,
        reason='paddle do not support arg "prepend"',
    )
