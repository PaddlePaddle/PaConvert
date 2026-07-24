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

obj = APIBase("torch.nn.Module.register_forward_hook")


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
                return x, x, x1
        def hook(module, fea_in, fea_out):
            result.append(1)

        net = TestForHook()
        net.register_forward_hook(hook)
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
                return x, x, x1
        def hook(module, fea_in, fea_out):
            result.append(1)

        net = TestForHook()
        net.register_forward_hook(hook=hook, prepend=False, with_kwargs=False, always_call=False)
        a = torch.tensor([0.,0.])
        net(a)
        """
    )
    obj.run(pytorch_code, ["result"])


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
                return x, x, x1
        def hook(module, fea_in, fea_out):
            result.append(1)

        net = TestForHook()
        net.register_forward_hook(prepend=False, hook=hook, always_call=False ,with_kwargs=False)
        a = torch.tensor([0.,0.])
        net(a)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    """prepend=True runs the new hook before existing hooks."""
    pytorch_code = textwrap.dedent(
        """
        import torch

        events = []

        class DoubleNum(torch.nn.Module):
            def forward(self, x):
                return 2 * x

        def first_hook(module, args, output):
            events.append("first")
            return output + 1

        def second_hook(module, args, output):
            events.append("second")
            return output * 2

        net = DoubleNum()
        net.register_forward_hook(second_hook)
        net.register_forward_hook(first_hook, prepend=True)
        x = torch.tensor(
            [[-1.5, 2.0], [0.25, -3.0]],
            dtype=torch.float64,
        )
        result = net(x)
        """
    )
    obj.run(pytorch_code, ["events", "result"])


def test_case_7():
    """with_kwargs=True passes forward keyword arguments to the hook."""
    pytorch_code = textwrap.dedent(
        """
        import torch

        observed_kwargs = []

        class ScaleAndShift(torch.nn.Module):
            def forward(self, x, scale=1.0, offset=0.0):
                return x * scale + offset

        def hook(module, args, kwargs, output):
            observed_kwargs.append((kwargs["scale"], kwargs["offset"]))
            return output + kwargs["scale"]

        net = ScaleAndShift()
        net.register_forward_hook(hook, with_kwargs=True)
        x = torch.tensor(
            [[[-1.0, 0.5], [2.0, -3.0]]],
            dtype=torch.float32,
        )
        result = net(x, scale=2.5, offset=-1.0)
        """
    )
    obj.run(pytorch_code, ["observed_kwargs", "result"])


def test_case_8():
    """always_call=True invokes the hook when forward raises."""
    pytorch_code = textwrap.dedent(
        """
        import torch

        events = []

        class BrokenModule(torch.nn.Module):
            def forward(self, x):
                raise RuntimeError("forward failed")

        def hook(module, args, output):
            events.append(output is None)

        net = BrokenModule()
        net.register_forward_hook(hook, always_call=True)
        try:
            net(torch.tensor([1.0, -2.0]))
        except RuntimeError:
            pass
        """
    )
    obj.run(pytorch_code, ["events"])


def test_case_9():
    """The returned handle removes the registered hook."""
    pytorch_code = textwrap.dedent(
        """
        import torch

        events = []

        class AddOne(torch.nn.Module):
            def forward(self, x):
                return x + 1

        def hook(module, args, output):
            events.append("called")
            return output * 2

        net = AddOne()
        handle = net.register_forward_hook(hook)
        result_before_remove = net(torch.tensor([1, -2, 3]))
        handle.remove()
        result_after_remove = net(torch.tensor([1, -2, 3]))
        """
    )
    obj.run(
        pytorch_code,
        ["events", "result_before_remove", "result_after_remove"],
    )


def test_case_10():
    """The hook can be supplied through variable positional arguments."""
    pytorch_code = textwrap.dedent(
        """
        import torch

        class DoubleNum(torch.nn.Module):
            def forward(self, x):
                return 2 * x

        def hook(module, args, output):
            return output - 1

        net = DoubleNum()
        hook_args = (hook,)
        net.register_forward_hook(*hook_args)
        x = torch.tensor([[[-2.0, 1.0], [0.5, -0.25]]])
        result = net(x)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_11():
    """All arguments can be supplied through variable keyword arguments."""
    pytorch_code = textwrap.dedent(
        """
        import torch

        events = []

        class DoubleNum(torch.nn.Module):
            def forward(self, x):
                return 2 * x

        def hook(module, args, kwargs, output):
            events.append(len(kwargs))
            return output + 3

        net = DoubleNum()
        hook_kwargs = {
            "hook": hook,
            "prepend": False,
            "with_kwargs": True,
            "always_call": False,
        }
        net.register_forward_hook(**hook_kwargs)
        result = net(torch.tensor([-1.0, 2.0]))
        """
    )
    obj.run(pytorch_code, ["events", "result"])
