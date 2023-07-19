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
#

#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import textwrap

from apibase import APIBase

obj = APIBase("torch.stft")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 2, 3], [3, 4, 6]])
        n_fft = 16
        result = torch.stft(x, n_fft=n_fft, return_complex=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 2, 3], [3, 4, 6]])
        n_fft = 16
        hop_length = 4
        result = torch.stft(x, n_fft=n_fft, hop_length=hop_length, return_complex=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 2, 3], [3, 4, 6]])
        n_fft = 16
        win_length = 16
        result = torch.stft(x, n_fft=n_fft, win_length=win_length, return_complex=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 2, 3], [3, 4, 6]])
        n_fft = 16
        result = torch.stft(x, n_fft=n_fft, center=False, return_complex=True)
        """
    )
    obj.run(pytorch_code, ["result"])


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        x = torch.tensor([[1, 2, 3], [3, 4, 6]])
        n_fft = 16
        result = torch.stft(x, n_fft=n_fft, center=False, onesided=False, return_complex=True)
        """
    )
    obj.run(pytorch_code, ["result"])
