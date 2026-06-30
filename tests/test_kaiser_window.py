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

import pytest
from apibase import APIBase

obj = APIBase("torch.kaiser_window")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        args = (7, False, 6.0)
        result = torch.kaiser_window(*args)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.kaiser_window(
            window_length=7,
            periodic=False,
            beta=6.0,
            dtype=torch.float64,
            layout=torch.strided,
            device="cpu",
        )
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.kaiser_window(7)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.kaiser_window(7, dtype=None)
        """
    )
    obj.run(pytorch_code, ["result"])


@pytest.mark.skip(reason="torch.kaiser_window does not accept out in current PyTorch")
def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        out = torch.empty(7)
        result = torch.kaiser_window(7, out=out)
        output = out
        """
    )
    obj.run(pytorch_code, ["result", "output"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result0 = torch.kaiser_window(0)
        result1 = torch.kaiser_window(1)
        """
    )
    obj.run(pytorch_code, ["result0", "result1"])


@pytest.mark.skip(reason="torch.sparse_coo conversion is not supported by PaConvert")
def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import pytest
        import torch

        with pytest.raises(RuntimeError):
            torch.kaiser_window(7, layout=torch.sparse_coo)
        """
    )
    obj.run(pytorch_code, [])


def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.kaiser_window(9, beta=8.5, periodic=True, dtype=torch.float32)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_9():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.kaiser_window(
            beta=5.5,
            device="cpu",
            window_length=11,
            dtype=torch.float64,
            periodic=False,
            layout=torch.strided,
        )
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_10():
    pytorch_code = textwrap.dedent(
        """
        import torch
        result = torch.kaiser_window(
            13,
            periodic=True,
            beta=7.0,
            dtype=torch.float64,
            layout=torch.strided,
            device="cpu",
        )
        """
    )
    obj.run(pytorch_code, ["result"])
