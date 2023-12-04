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

obj = APIBase("torch.autograd.grad")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1.1, 2.2, 3.3], requires_grad=True)
        y = x * x

        result = torch.autograd.grad([y.sum()], [x])[0]
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1.1, 2.2, 3.3], requires_grad=True)
        z = torch.tensor([1.1, 2.2, 3.3], requires_grad=True)
        y = x * x + z

        result = torch.autograd.grad([y.sum()], [x, z])
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1.1, 2.2, 3.3], requires_grad=True)
        z = torch.tensor([1.1, 2.2, 3.3], requires_grad=True)
        grad = torch.tensor(2.0)
        y = x * x + z

        result = torch.autograd.grad(outputs=[y.sum()], inputs=[x, z], grad_outputs=grad)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1.1, 2.2, 3.3], requires_grad=True)
        z = torch.tensor([1.1, 2.2, 3.3], requires_grad=True)
        grad = torch.tensor(2.0)
        y = x * x + z

        result = torch.autograd.grad(outputs=[y.sum()], inputs=[x, z], grad_outputs=grad)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1.1, 2.2, 3.3], requires_grad=True)
        z = torch.tensor([1.1, 2.2, 3.3], requires_grad=True)
        grad = torch.tensor(2.0)
        y = x * x + z

        result = torch.autograd.grad(outputs=[y.sum()], inputs=[x, z], grad_outputs=grad, retain_graph=True,
            create_graph=False, allow_unused=True, is_grads_batched=False)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1.1, 2.2, 3.3], requires_grad=True)
        z = torch.tensor([1.1, 2.2, 3.3], requires_grad=True)
        grad = torch.tensor(2.0)
        y = x * x + z

        result = torch.autograd.grad(outputs=[y.sum()], inputs=[x, z], grad_outputs=grad, retain_graph=True,
            create_graph=False, allow_unused=True, is_grads_batched=False, only_inputs=True)
        """
    )
    obj.run(
        pytorch_code,
        ["result"],
        unsupport=True,
        reason="paddle dose not support 'only_inputs' now!",
    )


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([1.1, 2.2, 3.3], requires_grad=True)
        z = torch.tensor([1.1, 2.2, 3.3], requires_grad=True)
        grad = torch.tensor(2.0)
        y = x * x + z

        result = torch.autograd.grad(allow_unused=True, inputs=[x, z], grad_outputs=grad, retain_graph=True,
            create_graph=False, outputs=[y.sum()], is_grads_batched=False)
        """
    )
    obj.run(pytorch_code, ["result"])
