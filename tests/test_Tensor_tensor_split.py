import textwrap, os

from apibase import APIBase

obj = APIBase("torch.Tensor.tensor_split")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.arange(8)
        result = a.tensor_split(4)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.arange(7)
        result = a.tensor_split(3)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.arange(7)
        result = a.tensor_split((1, 6))
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.arange(14).reshape(2, 7)
        result = a.tensor_split(3, dim = 1)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        a = torch.arange(14).reshape(2, 7)
        result = a.tensor_split((1, 6), dim = 1)
        """
    )
    obj.run(pytorch_code, ["result"])
