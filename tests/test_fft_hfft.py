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

obj = APIBase("torch.fft.hfft")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        t = torch.arange(5)
        t = torch.linspace(0, 1, 5)
        T = torch.fft.ifft(t)
        result = torch.fft.hfft(T[:3], n=5)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        t = torch.arange(5)
        t = torch.linspace(0, 1, 5)
        T = torch.fft.ifft(t)
        result = torch.fft.hfft(T[:3])
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        t = torch.arange(5)
        t = torch.linspace(0, 1, 5)
        T = torch.fft.ifft(t)
        result = torch.fft.hfft(T[:3], n=5, dim=0)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        t = torch.arange(5)
        t = torch.linspace(0, 1, 5)
        T = torch.fft.ifft(t)
        result = torch.fft.hfft(T[:3], n=5, norm='ortho')
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        t = torch.arange(5)
        t = torch.linspace(0, 1, 5)
        T = torch.fft.ifft(t)
        result = torch.fft.hfft(T[:3], n=5, norm='forward')
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        t = torch.arange(5)
        t = torch.linspace(0, 1, 5)
        T = torch.fft.ifft(t)
        result = torch.fft.hfft(T[:3], n=5, norm='backward')
        """
    )
    obj.run(pytorch_code, ["result"])
