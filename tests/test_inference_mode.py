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

obj = APIBase("torch.inference_mode")


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


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.ones(1, 2, 3, requires_grad=True)
        @torch.inference_mode(mode= False)
        def doubler(x):
            return x * 2
        result = (doubler(x).requires_grad, doubler(x))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    # explicit mode=True keyword argument form
    # Note: cast offset tensor to float32 so we don't rely on cross-framework
    #       implicit integer->floating type promotion behaviour.
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.ones(4, requires_grad=True)
        @torch.inference_mode(mode=True)
        def adder(x):
            y = x + torch.arange(4).to(torch.float32)
            z = y * (-0.5 + 0.05*10)
            return z.sum()
        out = adder(x)
        result = (out.requires_grad, int(out.item() == ((x.detach() + torch.arange(4).to(torch.float32)) * 4.5).sum().item()))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_7():
    # nested context managers; inner exit should restore outer state correctly
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.ones(3, requires_grad=True)
        with torch.inference_mode(True):
            a = x * 10
            with torch.inference_mode(False):
                b = x * 20
            c = x * 30
        result = (a.requires_grad, b.requires_grad, c.requires_grad,
                  bool((b.grad_fn is not None)),
                  int(a[0].item()), int(b[0].item()))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_8():
    # function decorated twice (stacked decorators) should remain inside inference scope
    pytorch_code = textwrap.dedent(
        """
        import torch

        @torch.inference_mode()
        @torch.inference_mode()
        def fn(t):
            return t.sin().cos()

        x = torch.linspace(-1., 1., steps=6, requires_grad=True)
        r = fn(x)
        result = (r.requires_grad,)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)
        with torch.inference_mode(mode=True):
            result = x * 2
        """
    )
    obj.run(pytorch_code, ["result"], check_stop_gradient=False)


def test_case_10():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)
        with torch.inference_mode(mode=False):
            result = x * 3
        """
    )
    obj.run(pytorch_code, ["result"], check_stop_gradient=False)


def test_case_11():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)
        @torch.inference_mode
        def doubler(y):
            return y * 2
        result = doubler(x)
        """
    )
    obj.run(pytorch_code, ["result"], check_stop_gradient=False)
