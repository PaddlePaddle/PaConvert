# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Licensed under Apache License 2.0

import textwrap

import pytest
from apibase import APIBase

obj = APIBase("torch.autograd.function.once_differentiable")


def _test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch

        class CustomFunction(torch.autograd.Function):
            @staticmethod
            @torch.autograd.function.once_differentiable
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return x * 2

            @staticmethod
            def backward(ctx, grad_output):
                x, = ctx.saved_tensors
                return grad_output * 3

        x = torch.tensor([1.0, 2.0], requires_grad=True)
        y = CustomFunction.apply(x)
        y.sum().backward()
        result_y = y
        result_grad = x.grad
    """
    )
    obj.run(pytorch_code, ["result_y", "result_grad"])


def _test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch

        class CustomFunction(torch.autograd.Function):
            @staticmethod
            @torch.autograd.function.once_differentiable
            def forward(ctx, a, b):
                ctx.save_for_backward(a, b)
                return a + b * 2

            @staticmethod
            def backward(ctx, grad_output):
                a, b = ctx.saved_tensors
                return grad_output, grad_output * 2

        a = torch.tensor([1.0, 2.0], requires_grad=True)
        b = torch.tensor([3.0, 4.0], requires_grad=True)
        y = CustomFunction.apply(a, b)
        y.sum().backward()
    """
    )
    obj.run(pytorch_code, ["y", "a.grad", "b.grad"])


def _test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch

        class SquareFunction(torch.autograd.Function):
            @staticmethod
            @torch.autograd.function.once_differentiable
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return x ** 2

            @staticmethod
            def backward(ctx, grad_output):
                x, = ctx.saved_tensors
                return grad_output * 2 * x

        x = torch.tensor([2.0, 3.0], requires_grad=True)
        y = SquareFunction.apply(x)
        z = y * 3
        z.sum().backward()
    """
    )
    obj.run(pytorch_code, ["y", "z", "x.grad"])


def _test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch

        class ComplexFunction(torch.autograd.Function):
            @staticmethod
            @torch.autograd.function.once_differentiable
            def forward(ctx, x, factor=2.0):
                ctx.factor = factor
                ctx.save_for_backward(x)
                return x * factor

            @staticmethod
            def backward(ctx, grad_output):
                x, = ctx.saved_tensors
                return grad_output * ctx.factor, None

        x = torch.tensor([1.0, 2.0], requires_grad=True)
        y = ComplexFunction.apply(x)
        y.sum().backward()
    """
    )
    obj.run(pytorch_code, ["y", "x.grad"])


def _test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch

        class DTFunction(torch.autograd.Function):
            @staticmethod
            @torch.autograd.function.once_differentiable
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return x.float() * 1.5

            @staticmethod
            def backward(ctx, grad_output):
                x, = ctx.saved_tensors
                return grad_output.to(x.dtype) * 1.5

        x = torch.tensor([1, 2], dtype=torch.int32, requires_grad=True)
        y = DTFunction.apply(x)
        y.sum().backward()
    """
    )
    obj.run(pytorch_code, ["y", "x.grad"])


def _test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch

        class NoGradFunction(torch.autograd.Function):
            @staticmethod
            @torch.autograd.function.once_differentiable
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return x + 1

            @staticmethod
            def backward(ctx, grad_output):
                x, = ctx.saved_tensors
                return grad_output

        x = torch.tensor([1.0, 2.0], requires_grad=False)
        y = NoGradFunction.apply(x)
        result_y = y
    """
    )
    obj.run(pytorch_code, ["result_y"])


def _test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch

        class NestedFunction(torch.autograd.Function):
            @staticmethod
            @torch.autograd.function.once_differentiable
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return torch.sin(x)

            @staticmethod
            def backward(ctx, grad_output):
                x, = ctx.saved_tensors
                return grad_output * torch.cos(x)

        x = torch.tensor([0.0, 1.0], requires_grad=True)
        y = NestedFunction.apply(x)
        z = y * y
        z.sum().backward()
    """
    )
    obj.run(pytorch_code, ["y", "z", "x.grad"])


@pytest.mark.skip(reason="Complex tensor operations require special handling")
def _test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch

        class ComplexTensorFunction(torch.autograd.Function):
            @staticmethod
            @torch.autograd.function.once_differentiable
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return x.conj()

            @staticmethod
            def backward(ctx, grad_output):
                x, = ctx.saved_tensors
                return grad_output.conj()

        x = torch.tensor([1+2j, 3+4j], requires_grad=True)
        y = ComplexTensorFunction.apply(x)
        y.sum().backward()
    """
    )
    obj.run(pytorch_code, ["y", "x.grad"])
