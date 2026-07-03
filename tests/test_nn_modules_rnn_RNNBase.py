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

import pytest
from apibase import APIBase

obj = APIBase("torch.nn.modules.rnn.RNNBase")


def test_case_1():
    pytorch_code = textwrap.dedent(
        """
        import torch
        model = torch.nn.modules.rnn.RNNBase('RNN_TANH', 10, 20, 2)
        """
    )
    obj.run(pytorch_code)


def test_case_2():
    pytorch_code = textwrap.dedent(
        """
        import torch
        model = torch.nn.modules.rnn.RNNBase(mode='RNN_TANH', input_size=10, hidden_size=20, num_layers=2)
        """
    )
    obj.run(pytorch_code)


def test_case_3():
    pytorch_code = textwrap.dedent(
        """
        import torch
        model = torch.nn.modules.rnn.RNNBase('LSTM', 10, 20, 2, bias=True, batch_first=False)
        """
    )
    obj.run(pytorch_code)


def test_case_4():
    pytorch_code = textwrap.dedent(
        """
        import torch
        model = torch.nn.modules.rnn.RNNBase('GRU', input_size=10, hidden_size=20, num_layers=2, bias=True, batch_first=True)
        """
    )
    obj.run(pytorch_code)


def test_case_5():
    pytorch_code = textwrap.dedent(
        """
        import torch
        model = torch.nn.modules.rnn.RNNBase('LSTM', 10, 20, 1, batch_first=True)
        """
    )
    obj.run(pytorch_code)


def test_case_6():
    pytorch_code = textwrap.dedent(
        """
        import torch
        model = torch.nn.modules.rnn.RNNBase('GRU', 10, 20, 2, dropout=0.2)
        """
    )
    obj.run(pytorch_code)


def test_case_7():
    pytorch_code = textwrap.dedent(
        """
        import torch
        model = torch.nn.modules.rnn.RNNBase('RNN_TANH', 10, 20, 2, bidirectional=True)
        """
    )
    obj.run(pytorch_code)


@pytest.mark.skip(
    reason="Paddle RNN does not support bias=False, causing 'list assignment index out of range'"
)
def test_case_8():
    pytorch_code = textwrap.dedent(
        """
        import torch
        model = torch.nn.modules.rnn.RNNBase('RNN_TANH', 10, 20, 2, bias=False)
        """
    )
    obj.run(pytorch_code)
