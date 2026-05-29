# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

obj = APIBase("torchaudio.functional.resample")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torchaudio
        waveform = torch.sin(torch.arange(1000, dtype=torch.float32) * 0.1).unsqueeze(0)
        result = torchaudio.functional.resample(waveform, 16000, 8000)
        """
    )
    obj.run(pytorch_code, ["result"], rtol=1e-4)


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torchaudio
        waveform = torch.sin(torch.arange(1000, dtype=torch.float32) * 0.1).unsqueeze(0)
        result = torchaudio.functional.resample(waveform=waveform, orig_freq=16000, new_freq=8000)
        """
    )
    obj.run(pytorch_code, ["result"], rtol=1e-4)


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torchaudio
        waveform = torch.sin(torch.arange(1000, dtype=torch.float32) * 0.1).unsqueeze(0)
        result = torchaudio.functional.resample(new_freq=8000, waveform=waveform, orig_freq=16000)
        """
    )
    obj.run(pytorch_code, ["result"], rtol=1e-4)


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torchaudio
        waveform = torch.sin(torch.arange(1000, dtype=torch.float32) * 0.1).unsqueeze(0)
        result = torchaudio.functional.resample(waveform, 16000, 8000, lowpass_filter_width=12, rolloff=0.95)
        """
    )
    obj.run(pytorch_code, ["result"], rtol=1e-4)


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torchaudio
        waveform = torch.sin(torch.arange(1000, dtype=torch.float32) * 0.1).unsqueeze(0)
        result = torchaudio.functional.resample(waveform, 8000, 16000, resampling_method="sinc_interp_kaiser", beta=12.0)
        """
    )
    obj.run(pytorch_code, ["result"], rtol=1e-4)


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        import torchaudio
        waveform = torch.sin(torch.arange(1000, dtype=torch.float32) * 0.1).unsqueeze(0).unsqueeze(0).expand(4, 1, -1)
        result = torchaudio.functional.resample(waveform, 44100, 16000)
        """
    )
    obj.run(pytorch_code, ["result"], rtol=1e-4)
